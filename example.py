from mfcc_implementation import mfcc_base
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc_base.mfcc(sig,rate)
d_mfcc_feat = mfcc_base.delta(mfcc_feat, 2)
fbank_feat = mfcc_base.logfbank(sig,rate)

print(fbank_feat[1:3,:])