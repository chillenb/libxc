
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_lc_tmlyp_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.215850332451930e+01, -2.215855687049793e+01, -2.215885834594829e+01, -2.215806208864061e+01, -2.215847026074100e+01, -2.215847026074100e+01, -3.351466011900868e+00, -3.351507011790574e+00, -3.352841445656810e+00, -3.354486051628999e+00, -3.353164713411350e+00, -3.353164713411350e+00, -5.211020535456717e-01, -5.206830354584776e-01, -5.128032002264871e-01, -5.243112924551893e-01, -5.207801571404727e-01, -5.207801571404727e-01, -7.094088355465548e-02, -7.349465834508476e-02, -5.156076015592089e-01, -3.025801801575304e-02, -4.614462801424009e-02, -4.614462801424010e-02, -1.055553972278456e-03, -1.110751371911792e-03, -1.324076820261277e-03, -6.110006192497843e-04, -7.666158941444470e-04, -7.666158941444470e-04, -5.315775871654441e+00, -5.316342326539671e+00, -5.315826853457939e+00, -5.316325976713594e+00, -5.316049756393332e+00, -5.316049756393332e+00, -1.989539682504291e+00, -2.006115690209546e+00, -1.986565917104059e+00, -2.001371433201749e+00, -2.000182435751154e+00, -2.000182435751154e+00, -4.837787829777160e-01, -5.313140258847638e-01, -4.298546976870357e-01, -4.428579316836074e-01, -4.938415090216927e-01, -4.938415090216927e-01, 9.427388747996063e-04, -6.572382773105423e-02, 3.341561699670752e-03, -1.776650765523115e+00, -1.367260507670045e-02, -1.367260507670045e-02, -4.719383075983923e-04, -5.972049510573814e-04, -4.571380411657196e-04, 4.549148270898235e-03, -5.500885636344090e-04, -5.500885636344090e-04, -4.958499385028194e-01, -4.919035149290661e-01, -4.932263863108964e-01, -4.943673800703974e-01, -4.937891874016748e-01, -4.937891874016748e-01, -4.776102640797085e-01, -3.983626806596429e-01, -4.197678839761381e-01, -4.414992211495063e-01, -4.301307316803914e-01, -4.301307316803914e-01, -5.581349019966786e-01, -1.119775943935951e-01, -1.568723098369333e-01, -2.378424808092277e-01, -1.957489144069365e-01, -1.957489144069364e-01, -3.511780252645881e-01, -1.674020414263933e-03, 3.470122622494087e-03, -2.255271308053624e-01, 1.641948852476221e-03, 1.641948852476205e-03, -1.490696290428996e-03, -1.606048732886797e-04, -3.361408055534047e-04, 2.526835601756408e-03, -5.092570901722635e-04, -5.092570901722630e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_lc_tmlyp_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.789272179835007e+01, -2.789284129484878e+01, -2.789333900033176e+01, -2.789156232484714e+01, -2.789249802677377e+01, -2.789249802677377e+01, -4.311205446267460e+00, -4.311304788343708e+00, -4.313834291701957e+00, -4.311664506918012e+00, -4.312089012501970e+00, -4.312089012501970e+00, -7.060530714187608e-01, -7.043692991844412e-01, -6.663246548147909e-01, -6.784356706472314e-01, -6.758771468050651e-01, -6.758771468050651e-01, -1.300040989503727e-01, -1.321056120611697e-01, -7.803614996950911e-01, -9.251506050866505e-02, -1.051423684201403e-01, -1.051423684201403e-01, -1.391555122047313e-03, -1.463367585454661e-03, -6.540946150639323e-03, -8.103861764145483e-04, -1.015234384729577e-03, -1.015234384729577e-03, -6.830973708193885e+00, -6.834302845769219e+00, -6.831125405479411e+00, -6.834064279619466e+00, -6.832662265531655e+00, -6.832662265531655e+00, -2.285841437583880e+00, -2.309257879774499e+00, -2.268107674303327e+00, -2.288803479927418e+00, -2.307646069041794e+00, -2.307646069041794e+00, -6.647650441369918e-01, -7.626157600294579e-01, -5.991104491709662e-01, -6.564365938781779e-01, -6.811804615270983e-01, -6.811804615270983e-01, -6.204575621720404e-02, -1.394355251261143e-01, -5.463479041508852e-02, -2.520854118360900e+00, -7.135998448180791e-02, -7.135998448180791e-02, -6.270971896446351e-04, -7.924133208066941e-04, -6.079758250793657e-04, -2.395808266790239e-02, -7.309028131489648e-04, -7.309028131489648e-04, -7.024088694077972e-01, -6.929486208495603e-01, -6.964063003824977e-01, -6.991678842660973e-01, -6.977989092549280e-01, -6.977989092549280e-01, -6.815207920267017e-01, -5.285021209512590e-01, -5.698686424703671e-01, -6.144738510125526e-01, -5.914645727653670e-01, -5.914645727653670e-01, -8.035220899789106e-01, -1.866431587182866e-01, -2.325992194179273e-01, -3.371198141499916e-01, -2.785238738963681e-01, -2.785238738963680e-01, -4.746533916300653e-01, -5.891783579647520e-03, -1.370813167011979e-02, -3.277094346521115e-01, -3.838103188822253e-02, -3.838103188822249e-02, -1.947516084489869e-03, -2.139595033893058e-04, -4.472809026493709e-04, -3.438608574617551e-02, -6.769311129669422e-04, -6.769311129669417e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.729833315081121e-09, -7.729759370208637e-09, -7.729336119829434e-09, -7.730435501078316e-09, -7.729872934640282e-09, -7.729872934640282e-09, -1.268531712444086e-05, -1.268510867696558e-05, -1.267539772476469e-05, -1.264306409371889e-05, -1.266505679297199e-05, -1.266505679297199e-05, -7.549826762287548e-03, -7.545591264530505e-03, -7.302779391036881e-03, -6.930982549457053e-03, -7.066709624544930e-03, -7.066709624544930e-03, -3.641244363565366e-02, -4.913209955937092e-02, -6.259311528793216e-03, 3.234767719066466e-01, 1.077169046146558e-01, 1.077169046146553e-01, -3.598513341145089e+00, -3.587402564066178e+00, 7.228076258563220e+00, -3.700303556217913e+00, -3.669887728067691e+00, -3.669887728067689e+00, -2.162184795204181e-06, -2.160813127939321e-06, -2.162075597574763e-06, -2.160866693637134e-06, -2.161515570577613e-06, -2.161515570577613e-06, -8.218227025033896e-05, -8.014463475857934e-05, -8.182238033981225e-05, -8.001652234742561e-05, -8.120865856370288e-05, -8.120865856370288e-05, -1.024199215546929e-02, -7.822334900573970e-03, -1.429804798979921e-02, -1.401914022176025e-02, -9.703111945115302e-03, -9.703111945115302e-03, 1.927963796450551e+00, 1.931985505737988e-02, 2.485303606937806e+00, -1.481690684372111e-04, 9.884554612985207e-01, 9.884554612985207e-01, -3.737244522198776e+00, -3.707889496941281e+00, -3.752677391421721e+00, 6.269929989178689e+00, -3.731075469384574e+00, -3.731075469384574e+00, -7.815678655994363e-03, -8.718943685683036e-03, -8.444712314442399e-03, -8.184388112552850e-03, -8.318800845765415e-03, -8.318800845765415e-03, -8.489185627382630e-03, -1.554388901845315e-02, -1.430103760245215e-02, -1.263906401401432e-02, -1.358270341582689e-02, -1.358270341582690e-02, -6.978726235652591e-03, -5.264379794711166e-02, -6.307992863030107e-02, -4.976381161337799e-02, -5.909527373160294e-02, -5.909527373160288e-02, -2.150329108887704e-02, 6.807612095596272e+00, 7.898506240979740e+00, -5.811932909751313e-02, 3.785801204078266e+00, 3.785801204078265e+00, -3.492487213804162e+00, -3.798751031535253e+00, -3.766095381825669e+00, 4.426118089062424e+00, -3.740098566464697e+00, -3.740098566464700e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.027064041634846e-04, 8.027097432317492e-04, 8.027149506552770e-04, 8.026654721003188e-04, 8.026927678336370e-04, 8.026927678336370e-04, 4.508477936483647e-03, 4.508727229113284e-03, 4.514387192866657e-03, 4.503709632962018e-03, 4.508669424453144e-03, 4.508669424453144e-03, 1.753822252068030e-02, 1.747885744085603e-02, 1.597139840956110e-02, 1.595512342473314e-02, 1.602759607535826e-02, 1.602759607535826e-02, 1.496670968698741e-02, 1.546176961836667e-02, 1.612815745533796e-02, 9.380504276344655e-03, 1.167834701695899e-02, 1.167834701695900e-02, 3.860835749153549e-06, 4.491310401312550e-06, 5.345232794065579e-04, 7.600121978878357e-07, 1.498693613866846e-06, 1.498693613866845e-06, 3.513438139925649e-03, 3.518172156308928e-03, 3.513627988598602e-03, 3.517807427501138e-03, 3.515851620917538e-03, 3.515851620917538e-03, 5.784161572269096e-03, 5.800238510629962e-03, 5.684949405872454e-03, 5.699343838478749e-03, 5.841828997289899e-03, 5.841828997289899e-03, 2.349098442457067e-02, 2.739241166571678e-02, 2.394413333066572e-02, 2.871702552944071e-02, 2.384701569402268e-02, 2.384701569402268e-02, 4.677038778457129e-03, 1.235126026702186e-02, 4.055431608071549e-03, 1.058786785563537e-02, 6.750161769586520e-03, 6.750161769586520e-03, 3.517078685740637e-07, 7.110520315397498e-07, 3.215520776399209e-07, 1.780336320258361e-03, 5.592275905120290e-07, 5.592275905120291e-07, 3.087713147954647e-02, 2.912393914260156e-02, 2.970681963986669e-02, 3.021611440758838e-02, 2.995834823652584e-02, 2.995834823652584e-02, 3.205091473600386e-02, 2.198611708588382e-02, 2.423581180590340e-02, 2.693717996190812e-02, 2.553244288784097e-02, 2.553244288784097e-02, 2.623935780428354e-02, 1.541517614376471e-02, 1.854827839622558e-02, 2.470139742091857e-02, 2.166824627523820e-02, 2.166824627523820e-02, 2.287248900030466e-02, 4.742836354764261e-04, 1.032381544724883e-03, 2.759138511626219e-02, 3.144600086977937e-03, 3.144600086977937e-03, 1.057116639194488e-05, 1.384902573632118e-08, 1.272834689793548e-07, 2.732151220570721e-03, 4.440234606080811e-07, 4.440234606080803e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05