
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hse12s_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse12s", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.213645491790247e+01, -1.213646765149120e+01, -1.213656979773458e+01, -1.213636038249377e+01, -1.213646125417123e+01, -1.213646125417123e+01, -2.093081760494251e+00, -2.093066428645236e+00, -2.092727910764308e+00, -2.093735960459123e+00, -2.093082398864223e+00, -2.093082398864223e+00, -4.804407087247829e-01, -4.802737037204293e-01, -4.765723235909450e-01, -4.793665714015030e-01, -4.803785514262628e-01, -4.803785514262628e-01, -1.795513135911638e-01, -1.807461834608146e-01, -5.527054472372341e-01, -1.507510159279059e-01, -1.798974664730047e-01, -1.798974664730047e-01, -1.703351862561000e-02, -1.783988660733185e-02, -6.834176748983747e-02, -8.177931618021465e-03, -1.765443369673201e-02, -1.765443369673201e-02, -2.998100820037957e+00, -2.997899694512712e+00, -2.998083448609255e+00, -2.997926985300267e+00, -2.997992641729925e+00, -2.997992641729925e+00, -1.293405004084769e+00, -1.299521909589472e+00, -1.293413265952562e+00, -1.298187516708077e+00, -1.297505228712198e+00, -1.297505228712198e+00, -4.187909638933736e-01, -4.465137924485196e-01, -4.017915806888100e-01, -4.136636901625150e-01, -4.310403536559611e-01, -4.310403536559611e-01, -1.252161921891762e-01, -1.882708294415436e-01, -1.235366226428786e-01, -1.161852082650635e+00, -1.362750799355641e-01, -1.362750799355641e-01, -7.889261528114510e-03, -9.015584565185060e-03, -6.757998158369241e-03, -8.870936788611805e-02, -8.214962546177309e-03, -8.214962546177309e-03, -4.303211815627408e-01, -4.267833889071734e-01, -4.279350211190042e-01, -4.289103003791714e-01, -4.284125772633403e-01, -4.284125772633403e-01, -4.195760815942326e-01, -3.774377773828274e-01, -3.871299037384360e-01, -3.972480390855081e-01, -3.918473028745591e-01, -3.918473028745591e-01, -4.638519201256613e-01, -2.175941364298737e-01, -2.421351710306917e-01, -2.853746876788986e-01, -2.624173230151891e-01, -2.624173230151891e-01, -3.481339543126327e-01, -6.430507855494545e-02, -8.217860797373665e-02, -2.713607439217664e-01, -1.071829704968611e-01, -1.071829704968611e-01, -1.995454396155206e-02, -2.468894622080688e-03, -4.689772072982388e-03, -1.030869763425475e-01, -6.982047698425278e-03, -6.982047699933911e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hse12s_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse12s", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.444715915575219e+01, -1.444721503366090e+01, -1.444745911549375e+01, -1.444654422284110e+01, -1.444718866732202e+01, -1.444718866732202e+01, -2.443237974808623e+00, -2.443266298722264e+00, -2.444141006736719e+00, -2.443121580587988e+00, -2.443262297344437e+00, -2.443262297344437e+00, -5.587954357865808e-01, -5.582048012017893e-01, -5.436560025311594e-01, -5.472497067084650e-01, -5.585800034302077e-01, -5.585800034302077e-01, -1.992096353852446e-01, -2.029942139531695e-01, -6.546799935617250e-01, -1.583617460615007e-01, -2.003600310640766e-01, -2.003600310640766e-01, -2.269932022502153e-02, -2.377243984034841e-02, -8.171516597642839e-02, -1.090275261166358e-02, -2.352568936581583e-02, -2.352568936581583e-02, -3.715356906290877e+00, -3.717465495208394e+00, -3.715571095627282e+00, -3.717210658694717e+00, -3.716440024939219e+00, -3.716440024939219e+00, -1.409116039025160e+00, -1.417765127697599e+00, -1.405976800943589e+00, -1.412773605152085e+00, -1.418918641583045e+00, -1.418918641583045e+00, -5.156035948842802e-01, -5.716642183819535e-01, -4.928792521289813e-01, -5.268149048771632e-01, -5.348963563988191e-01, -5.348963563988191e-01, -1.379938417700001e-01, -1.939533828108163e-01, -1.357883688058298e-01, -1.508572143464610e+00, -1.462092267489932e-01, -1.462092267489932e-01, -1.051798148024902e-02, -1.201921557732842e-02, -9.010035148704959e-03, -1.025204112071449e-01, -1.095212009062558e-02, -1.095212009062557e-02, -5.485809219946491e-01, -5.458481994266351e-01, -5.471720513819098e-01, -5.479497301072009e-01, -5.475899070981947e-01, -5.475899070981947e-01, -5.336851853767101e-01, -4.497782968839652e-01, -4.734038677952574e-01, -4.992818254709505e-01, -4.859792254072517e-01, -4.859792254072517e-01, -5.942095766903558e-01, -2.297125473819182e-01, -2.745165124678793e-01, -3.405733049143370e-01, -3.092732183363104e-01, -3.092732183363104e-01, -4.144515353208157e-01, -7.837959613801801e-02, -9.625113411174284e-02, -3.287384586232273e-01, -1.198097522912152e-01, -1.198097522912152e-01, -2.658504283127006e-02, -3.291829054627154e-03, -6.252818853714165e-03, -1.153173609815583e-01, -9.308696736074325e-03, -9.308696509456191e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hse12s_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse12s", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.305928255843748e-09, -4.305888144083130e-09, -4.305676952524416e-09, -4.306334372420656e-09, -4.305907371882031e-09, -4.305907371882031e-09, -5.071101860562160e-06, -5.071081889492064e-06, -5.069696569351599e-06, -5.067939777705728e-06, -5.071009303572392e-06, -5.071009303572392e-06, -1.951050916488412e-03, -1.952617180877589e-03, -1.956654055642186e-03, -1.912950547128677e-03, -1.951667057949042e-03, -1.951667057949042e-03, -1.978836048390572e-01, -1.792982429271410e-01, -1.078437554765417e-03, -4.669840591072726e-01, -1.922013060505658e-01, -1.922013060505658e-01, 2.948719632297825e-02, 3.177314056001308e-02, -2.931200915350037e+00, 5.521155581012880e-03, 3.311676184149787e-02, 3.311676184149787e-02, -9.968029176266807e-07, -9.937662869951013e-07, -9.964973291956223e-07, -9.941364477291540e-07, -9.952454092969178e-07, -9.952454092969178e-07, -3.364620666262454e-05, -3.315879433502223e-05, -3.344152494960840e-05, -3.305557888205271e-05, -3.360682771729806e-05, -3.360682771729806e-05, -2.411748211768715e-03, 1.782763718058302e-03, -3.023100231451007e-03, 1.022130304817152e-03, -1.812676933222308e-03, -1.812676933222308e-03, -6.421983063124517e-01, -1.803329619595459e-01, -7.287351764924580e-01, -5.415440813248752e-06, -6.129959752815934e-01, -6.129959752815934e-01, 5.689933158712631e-03, 7.087435648747788e-03, 2.145382733561061e-02, -1.767756896210706e+00, 1.094795505527167e-02, 1.094795505555073e-02, 4.967859210501488e-03, 2.716450235368232e-03, 3.448826303511968e-03, 4.070400696370628e-03, 3.753332288291851e-03, 3.753332288292036e-03, 6.174463821106485e-03, -5.064099470172667e-03, -3.682435454291538e-03, -1.162459827979358e-03, -2.592212071100226e-03, -2.592212071100299e-03, 1.230608755278764e-03, -8.783739865494260e-02, -3.905743448605142e-02, -1.577118165053136e-02, -2.168503543307378e-02, -2.168503543307400e-02, -7.071537139766349e-03, -2.581004953189556e+00, -1.868763318532374e+00, -1.830954799868631e-02, -1.228942351614936e+00, -1.228942351614941e+00, 2.438024817906253e-02, 1.779839569319765e-03, 4.326607523398382e-03, -1.527650500324427e+00, 1.480927522127662e-02, 1.480927522044869e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05