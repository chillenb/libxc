
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_1a_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.715517289572703e+01, -1.715520438569791e+01, -1.715539003836758e+01, -1.715487348299833e+01, -1.715518912489624e+01, -1.715518912489624e+01, -2.840401883778620e+00, -2.840392358951408e+00, -2.840247000291171e+00, -2.841095267900627e+00, -2.840408988860606e+00, -2.840408988860606e+00, -5.719478993631888e-01, -5.717359050026010e-01, -5.675341621247001e-01, -5.712825208199387e-01, -5.718686978139443e-01, -5.718686978139443e-01, -1.728984827727850e-01, -1.742576225246433e-01, -6.718937553188066e-01, -1.369755945871402e-01, -1.732939211533129e-01, -1.732939211533129e-01, -9.782316564805042e-03, -1.030010949047288e-02, -4.861336439634337e-02, -4.349062441608015e-03, -1.018028602876967e-02, -1.018028602876967e-02, -4.173498393948149e+00, -4.173665531210280e+00, -4.173519172561214e+00, -4.173649068557705e+00, -4.173578782130399e+00, -4.173578782130399e+00, -1.683438865654131e+00, -1.692239317337070e+00, -1.682879279701126e+00, -1.689702729670901e+00, -1.690236070265548e+00, -1.690236070265548e+00, -4.838882850196133e-01, -5.088653835928694e-01, -4.607419134660460e-01, -4.678818479063802e-01, -4.999052174424544e-01, -4.999052174424544e-01, -1.075788702397935e-01, -1.840850118409279e-01, -1.055067158033351e-01, -1.529511993637012e+00, -1.198816046853111e-01, -1.198816046853111e-01, -4.181419285120437e-03, -4.839437311658886e-03, -3.531494831969696e-03, -6.816875228721375e-02, -4.370299886624745e-03, -4.370299886624747e-03, -4.784051828632567e-01, -4.803337152950395e-01, -4.796399055758090e-01, -4.790767518982483e-01, -4.793580529100284e-01, -4.793580529100284e-01, -4.625941889581075e-01, -4.296684664542446e-01, -4.408259276612047e-01, -4.504092322469320e-01, -4.455745158569622e-01, -4.455745158569622e-01, -5.338785841662776e-01, -2.222753865438035e-01, -2.534059031334670e-01, -3.054504551975419e-01, -2.775911343316204e-01, -2.775911343316205e-01, -3.898934577841532e-01, -4.493322835039539e-02, -6.165370828991443e-02, -2.846049548332989e-01, -8.730739909953174e-02, -8.730739909953172e-02, -1.167767590013582e-02, -1.198665816030339e-03, -2.378130190691551e-03, -8.285995490513730e-02, -3.659269821099024e-03, -3.659269821099008e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_1a_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.182304283916258e+01, -2.182311178701436e+01, -2.182343521342105e+01, -2.182230583498131e+01, -2.182307906757097e+01, -2.182307906757097e+01, -3.529254444342163e+00, -3.529282335375072e+00, -3.530192603703894e+00, -3.529367525441864e+00, -3.529283752358114e+00, -3.529283752358114e+00, -6.757443010477410e-01, -6.747502356819027e-01, -6.474429895491742e-01, -6.529430168399964e-01, -6.753835552306044e-01, -6.753835552306044e-01, -1.925729968479051e-01, -1.952240997023281e-01, -8.159171817613491e-01, -1.441217588003420e-01, -1.933720395345488e-01, -1.933720395345488e-01, -1.336872939519467e-02, -1.407382490939818e-02, -6.236396783862671e-02, -5.932175901414042e-03, -1.390951826752078e-02, -1.390951826752078e-02, -5.361111898428423e+00, -5.362756151218978e+00, -5.361281632404440e+00, -5.362560133091596e+00, -5.361953303906689e+00, -5.361953303906689e+00, -1.892063279721123e+00, -1.908888790964686e+00, -1.881456971375937e+00, -1.894618313391058e+00, -1.917118695906813e+00, -1.917118695906813e+00, -5.999717669004575e-01, -6.562972519416127e-01, -5.693796423992100e-01, -5.963026740845211e-01, -6.234088551685462e-01, -6.234088551685462e-01, -1.175422553976608e-01, -1.914421636698037e-01, -1.151217984550673e-01, -1.994289133413182e+00, -1.273489761749619e-01, -1.273489761749619e-01, -5.702107467202400e-03, -6.604853425888303e-03, -4.809651614247053e-03, -8.210253908987332e-02, -5.960423900297571e-03, -5.960423900297569e-03, -6.308118515211018e-01, -6.208924525997667e-01, -6.237935028607980e-01, -6.265243983267366e-01, -6.250998430278860e-01, -6.250998430278860e-01, -6.121939111323751e-01, -5.171228664616947e-01, -5.431360976903480e-01, -5.659662573127566e-01, -5.544864905032629e-01, -5.544864905032629e-01, -6.875794820018389e-01, -2.366156101683066e-01, -2.813582827963421e-01, -3.635016599029063e-01, -3.201011293210276e-01, -3.201011293210276e-01, -4.670188387205735e-01, -5.850935945847868e-02, -7.648750816033277e-02, -3.453975715269844e-01, -9.852107840071016e-02, -9.852107840071012e-02, -1.595796843747053e-02, -1.621753283985824e-03, -3.231344570641444e-03, -9.386135142337787e-02, -4.985108345429953e-03, -4.985108345429943e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_1a_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.631571240540146e-09, -2.631539193730041e-09, -2.631386511884198e-09, -2.631911509459703e-09, -2.631554421332526e-09, -2.631554421332526e-09, -4.008521849971898e-06, -4.008491599294382e-06, -4.007001634891915e-06, -4.006152521003036e-06, -4.008437444126875e-06, -4.008437444126875e-06, -2.818056947087165e-03, -2.824189219332753e-03, -3.005062365400153e-03, -2.921056343517020e-03, -2.820307010993796e-03, -2.820307010993796e-03, -2.906774446326438e-01, -2.829093127326015e-01, -1.458297896171369e-03, -6.048465304419933e-01, -2.883847643988309e-01, -2.883847643988309e-01, -2.398964308073344e+00, -2.465235239910983e+00, -1.803509636462843e+00, -1.109091363639471e+00, -2.545368218130168e+00, -2.545368218130168e+00, -7.952072942893006e-07, -7.949868152222550e-07, -7.951816651308136e-07, -7.950101979661078e-07, -7.950970056233899e-07, -7.950970056233899e-07, -4.005969035768157e-05, -3.897691358900739e-05, -4.049292131718569e-05, -3.963604941718483e-05, -3.877512322076965e-05, -3.877512322076965e-05, -6.054568448038708e-03, -7.448082079985838e-03, -7.318001200138147e-03, -9.567249607128951e-03, -5.478931164250228e-03, -5.478931164250228e-03, -9.205350510885609e-01, -2.320280483595094e-01, -1.020236181417885e+00, -5.938520044792925e-05, -8.407327848329617e-01, -8.407327848329617e-01, -1.149922911104169e+00, -1.252587541172175e+00, -2.972938888916300e+00, -1.762351786860909e+00, -1.742687429483504e+00, -1.742687429489154e+00, -9.080720035140783e-03, -9.853029437407361e-03, -1.017913554941279e-02, -1.014470330011458e-02, -1.021051221081126e-02, -1.021051221081126e-02, -9.004208765451847e-03, -8.993295220710349e-03, -8.684012296977826e-03, -9.394142457342474e-03, -8.890141731892533e-03, -8.890141731892533e-03, -5.905236987941330e-03, -1.184875581144607e-01, -7.124998585214541e-02, -3.510237058367441e-02, -4.980976206452584e-02, -4.980976206452586e-02, -1.326091020164374e-02, -1.528039453337582e+00, -1.569674919644693e+00, -5.023423895403289e-02, -1.543211339509716e+00, -1.543211339509721e+00, -1.967291093892964e+00, -8.786228600941771e-01, -1.209111502158052e+00, -1.838685060867312e+00, -2.291455205030426e+00, -2.291455205035746e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05