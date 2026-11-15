// Fixed-policy inner replay only. Outer execution stays in the audited Python engine.
// No fast-math: preserve funded branch decisions and cash PnL arithmetic.
#include <cmath>
#include <algorithm>

extern "C" int equity_replay(const double* x, const double* close, const double* f, const double* evidence, const double* soft,
 const int* calendar, int contracts, int days, int start, int count, int policy,
 int variant, int bps, int overlay, int reentry, int regime, int evidence_model, int soft_exit, int evidence_count, double capital, double fee, double margin, double floor,
 double* out, double* trace) {
    const int P=4, F=15;
    int held=-1, code=-1, side=0, opened=-1, mean_streak=0, tail_streak=0;
    int trades=0, active=0, forced_count=0, demand=0, blocked=0, breaches=0;
    double equity=capital, peak=capital, dd=0, mark=0, entry_equity=capital, fees=0;
    double profit_peak=0.;
    int soft_streak=0,entry_limit=20;
    bool force_flat=false, halted=false;
    int last_exit[4][2];
    for(int p=0;p<4;++p) for(int s=0;s<2;++s) last_exit[p][s]=-100000;
    if (capital<=0 || start<0 || start+count>days) return 1;
    auto row=[&](int d,int p){return x+(d*P+p)*F;};
    auto reserve=[&](const double* r,int s,double cash){
        double adverse=std::max(.05,3*r[5]);
        double stressed=r[1]*(1-s*adverse);
        return cash-r[1]*adverse*r[4]-stressed*margin*r[4]>=.20*cash;
    };
    for(int k=0;k<count;++k){
        int d=start+k, wanted=-1, sign=side, raw=0, funded=0;
        if(reentry==2){
            for(int p=0;p<P;++p){
                const double* r=row(d,p);
                if(r[0]==0 || !std::isfinite(r[6]) || !std::isfinite(r[7]) || !std::isfinite(r[8])) continue;
                int si=r[7]>r[8]?1:0;
                bool valid=std::max(r[7],r[8])>.45 && std::abs(r[7]-r[8])>=.10;
                for(int s=0;s<2;++s) if(last_exit[p][s]>=0 && d>last_exit[p][s] && !(valid && s==si)) last_exit[p][s]=-100000;
            }
        }
        double liquidatable=equity;
        if(held>=0){
            double cn=close[d*contracts+code]*row(d,held)[4];
            liquidatable+=side*(cn-mark)-cn*fee;
        }
        if(overlay==3 || overlay==4) halted=halted || liquidatable<=floor;
        if(!halted && held>=0){
            const double* r=row(d,held);
            double previous_close=close[d*contracts+code]*r[4];
            if(!std::isfinite(previous_close)) return 2;
            double close_pnl=equity-entry_equity+side*(previous_close-mark)-previous_close*fee;
            profit_peak=std::max(profit_peak,close_pnl);
            bool exit=d-opened>=20 || (policy==3 && (side>0?r[10]:r[9])>=.35);
            if((overlay==2 || overlay==4) && close_pnl<=-.05*entry_equity) exit=true;
            if(!exit && soft_exit){
                if((soft_exit==3 || soft_exit==4 || soft_exit==7 || soft_exit==8) && d-opened>=entry_limit) exit=true;
                double giveback=(soft_exit==5 || r[14]==side)? .5:.25;
                if((soft_exit==5 || soft_exit==6 || soft_exit==7) && profit_peak>=.05*entry_equity && close_pnl<=profit_peak*(1-giveback)) exit=true;
                if(!exit && (soft_exit==1 || soft_exit==2 || soft_exit==7)){
                    const double* pred=soft+(((((opened*21+d-opened)*P+held)*2+bps)*2+(soft_exit==1?0:1))*2);
                    double utility=pred[0]-.5*pred[1];
                    soft_streak=(std::isfinite(utility) && utility<=0)?soft_streak+1:0;
                    exit=soft_streak>=2;
                }
            }
            if(!exit && (regime==2 || regime==3)){
                const double* state=f+((((opened*21+d-opened)*P+held)*2+bps)*5);
                if(state[4]!=side) return 3;
                exit=r[14]!=side && std::isfinite(state[0]) && state[0]<=0;
            }
            if(!exit && variant>=2){
                int age=d-opened;
                const double* state=f+((((opened*21+age)*P+held)*2+bps)*5);
                if(state[4]!=side) return 3;
                double net=variant==2?state[0]:state[1];
                mean_streak=(std::isfinite(net)&&net<=0)?mean_streak+1:0;
                double close_equity=equity+side*(previous_close-mark);
                double tail_fraction=close_equity>0?state[2]*state[3]/close_equity:INFINITY;
                tail_streak=(std::isfinite(state[2])&&tail_fraction>.05)?tail_streak+1:0;
                exit=mean_streak>=2 || (variant==4 && tail_streak>=2);
            }
            if(!exit && variant==1 && close_pnl<=-.05*entry_equity) exit=true;
            if(!exit && reserve(r,side,equity)) wanted=held;
        } else if(!halted && calendar[d]){
            double best=-INFINITY;
            for(int p=0;p<P;++p){
                const double* r=row(d,p);
                if(r[0]==0 || !std::isfinite(r[6])) continue;
                int s=r[7]>r[8]?1:-1;
                double confidence=std::max(r[7],r[8]);
                if(confidence<=.45 || std::abs(r[7]-r[8])<.10) continue;
                if(policy==2 && confidence<.60) continue;
                if(policy==1 && (s>0?r[11]:r[12])<.55) continue;
                ++raw;
                double n=r[1]*r[4];
                if(equity<n*margin+n*fee || !reserve(r,s,equity-n*fee)) continue;
                ++funded;
                if((regime==1 || regime==3) && r[14]==-s) continue;
                if(evidence_model){
                    const double* pred=evidence+((((((d*P+p)*2+(s>0?1:0))*2+bps)*2+(policy==3))*evidence_count+evidence_model-1)*3);
                    if(std::isfinite(pred[0]) && !(pred[0]>=.55 && std::isfinite(pred[1]) && std::isfinite(pred[2]) && pred[1]-.5*pred[2]>0)) continue;
                }
                if((overlay==1 || overlay==4) && (!std::isfinite(r[5]) || r[5]<0 || 2*r[5]*n>.05*equity)) continue;
                int previous=last_exit[p][s>0?1:0];
                if((reentry==1 && d-previous<=5) || (reentry==2 && previous>=0)) continue;
                if(confidence>best){best=confidence;wanted=p;sign=s;}
            }
            if(raw>0){++demand;if(funded==0) ++blocked;}
            if(wanted>=0){
                opened=d;entry_equity=equity;mean_streak=tail_streak=soft_streak=0;profit_peak=0.;
                int state=(int)row(d,wanted)[14];
                entry_limit=soft_exit==8?5:soft_exit==3?10:(soft_exit==4 || soft_exit==7)?(state==sign?20:state==0?10:5):20;
            }
        }
        bool forced=force_flat || (held>=0 && equity<mark*margin);
        if(forced){wanted=-1;++forced_count;}
        int newcode=wanted>=0?(int)row(d,wanted)[13]:-1;
        auto cost=[&](){
            if(held>=0 && wanted>=0 && code==newcode && side==sign) return 0.;
            double value=held>=0?mark*fee:0.;
            if(wanted>=0) value+=row(d,wanted)[1]*row(d,wanted)[4]*fee;
            return value;
        };
        double costs=cost();
        if(wanted>=0 && equity<row(d,wanted)[1]*row(d,wanted)[4]*margin+costs){
            wanted=-1;newcode=-1;costs=cost();
        }
        double gross=0;
        force_flat=false;
        if(wanted>=0){
            const double* r=row(d,wanted);
            if(held<0 || held!=wanted || side!=sign) ++trades;
            ++active;
            gross=sign*(r[2]-r[1])*r[4];
            force_flat=equity-costs+sign*(r[3]-r[1])*r[4]<r[3]*r[4]*margin;
            if(force_flat) ++breaches;
        }
        double pnl=gross-costs;
        equity+=pnl;
        if(held>=0 && (wanted!=held || sign!=side)) last_exit[held][side>0?1:0]=d;
        held=wanted;code=newcode;side=sign;
        mark=held>=0?row(d,held)[2]*row(d,held)[4]:0.;
        if(k==count-1 && held>=0){double closing=mark*fee;equity-=closing;pnl-=closing;costs+=closing;}
        fees+=costs;
        peak=std::max(peak,equity);
        dd=std::min(dd,equity/peak-1);
        if(trace){
            double* t=trace+k*10;
            t[0]=pnl;t[1]=equity;t[2]=held>=0?1:0;t[3]=costs;t[4]=forced;t[5]=force_flat;
            t[6]=raw;t[7]=funded;t[8]=code;t[9]=held>=0?side:0;
        }
    }
    out[0]=equity/capital-1;out[1]=dd;out[2]=trades;out[3]=active;out[4]=forced_count;
    out[5]=demand;out[6]=blocked;out[7]=breaches;out[8]=fees;
    return 0;
}
