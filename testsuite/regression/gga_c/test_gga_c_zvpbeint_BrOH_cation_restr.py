
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbeint_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.813844524758258e-02, -5.813930585568048e-02, -5.814175327549402e-02, -5.812897857766851e-02, -5.813586786713665e-02, -5.813586786713665e-02, -4.771139327825140e-02, -4.771658716469918e-02, -4.783690005756786e-02, -4.762828835923723e-02, -4.771765923021744e-02, -4.771765923021744e-02, -3.207374034235023e-02, -3.183954089671876e-02, -2.611811169159964e-02, -2.639208401837534e-02, -2.654102557212355e-02, -2.654102557212355e-02, -8.275809009787048e-03, -8.961959562870365e-03, -3.481104765701350e-02, -2.854012679047974e-03, -4.824516392174554e-03, -4.824516392174545e-03, -8.902311237825096e-09, -1.189721717799176e-08, -1.334978503351761e-05, -6.360139218202970e-10, -2.339563439500191e-09, -2.339563439500191e-09, -6.511881627022043e-02, -6.532554859909571e-02, -6.512733751319791e-02, -6.530983918088955e-02, -6.522400279216298e-02, -6.522400279216298e-02, -2.581248997172998e-02, -2.631870769446385e-02, -2.468256736077515e-02, -2.511869612748069e-02, -2.665800491726884e-02, -2.665800491726884e-02, -4.269294178305670e-02, -5.787299916340225e-02, -4.008702292415259e-02, -5.352868008118471e-02, -4.446583798153785e-02, -4.446583798153785e-02, -6.011872577899001e-04, -4.450473890048247e-03, -4.709733259713340e-04, -7.551878531061713e-02, -1.545844674160428e-03, -1.545844674160428e-03, -2.486014891759066e-10, -6.733547638466576e-10, -1.201817420545883e-09, -1.235471776815351e-04, -1.340479202383523e-09, -1.340479202383523e-09, -6.120263825876041e-02, -5.675816747997222e-02, -5.825611684107358e-02, -5.954784806035126e-02, -5.889592483917586e-02, -5.889592483917586e-02, -6.199230723157322e-02, -3.135329704173338e-02, -3.849855134934133e-02, -4.688385605822962e-02, -4.252761164803526e-02, -4.252761164803526e-02, -5.797478476423434e-02, -7.675425791203944e-03, -1.276490409071868e-02, -2.717458701288227e-02, -1.930010826256273e-02, -1.930010826256274e-02, -3.051277929226779e-02, -1.005392821624228e-05, -3.834722118832056e-05, -3.182370045387799e-02, -4.000680792445173e-04, -4.000680792445138e-04, -3.460000668700318e-08, -6.295661623528936e-12, -1.053010228046186e-10, -3.059876792321730e-04, -1.176706561052163e-09, -1.176706563220567e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbeint_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.354873133647907e-01, -1.354882736739634e-01, -1.354910164755558e-01, -1.354767610053696e-01, -1.354844474290627e-01, -1.354844474290627e-01, -1.160277953107055e-01, -1.160329748305725e-01, -1.161529345472292e-01, -1.159461644420918e-01, -1.160342337615511e-01, -1.160342337615511e-01, -8.311682127802927e-02, -8.290348406394323e-02, -7.655486285406847e-02, -7.698453850863370e-02, -7.715142539105686e-02, -7.715142539105686e-02, -3.395106929975191e-02, -3.587447689965560e-02, -8.756319832023265e-02, -1.449620239755738e-02, -2.247813085109308e-02, -2.247813085109308e-02, -5.809796134658723e-08, -7.760543651303066e-08, -8.450862434172727e-05, -4.169007030331714e-09, -1.530902001203561e-08, -1.530902001609553e-08, -1.345464751600681e-01, -1.347000360729234e-01, -1.345528368690693e-01, -1.346884112495193e-01, -1.346246977101890e-01, -1.346246977101890e-01, -8.355571518864356e-02, -8.448666396482406e-02, -8.146334204536392e-02, -8.229869272471865e-02, -8.507777883395477e-02, -8.507777883395477e-02, -8.498159571086986e-02, -8.211850666352401e-02, -8.281326582008050e-02, -8.013126389368358e-02, -8.538798341121190e-02, -8.538798341121190e-02, -3.487353117556771e-03, -2.128706524052947e-02, -2.763667912570134e-03, -1.172435977347532e-01, -8.390621560792105e-03, -8.390621560792105e-03, -1.632373532288945e-09, -4.414400888031607e-09, -7.891490357192085e-09, -7.570156725381764e-04, -8.791780197364420e-09, -8.791780199799526e-09, -7.518760817972248e-02, -7.929466770224376e-02, -7.805506107665536e-02, -7.686886666575057e-02, -7.748117552483347e-02, -7.748117552483347e-02, -7.279490800638030e-02, -7.769401292953780e-02, -8.134350719401115e-02, -8.179771178108967e-02, -8.203718754848199e-02, -8.203718754848199e-02, -8.410025168737263e-02, -3.291810988417814e-02, -4.703152048740811e-02, -6.904628322703699e-02, -5.955546815690029e-02, -5.955546815690030e-02, -7.573544926898737e-02, -6.379140022350110e-05, -2.398735908014121e-04, -7.006642236314210e-02, -2.362102826221014e-03, -2.362102826221021e-03, -2.250681271722518e-07, -4.157393504682863e-11, -6.928299115785861e-10, -1.825775686164231e-03, -7.721549877035000e-09, -7.721549883764276e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbeint_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.955365005848543e-10, 1.955398085119754e-10, 1.955440257670874e-10, 1.954950368047989e-10, 1.955221935353657e-10, 1.955221935353657e-10, 1.084799072775507e-06, 1.085005226678209e-06, 1.089656986013230e-06, 1.080588195521501e-06, 1.084918751424336e-06, 1.084918751424336e-06, 1.779327442094615e-03, 1.768782106676943e-03, 1.492926452600661e-03, 1.456494523039179e-03, 1.481468257795419e-03, 1.481468257795419e-03, 1.880813786325252e-01, 1.956950442502571e-01, 9.569870044370294e-04, 1.569356858634518e-01, 1.901129225729430e-01, 1.901129225729431e-01, 7.010242677205230e-03, 8.258909724247535e-03, 4.194561449561721e-02, 2.477514014653287e-03, 5.177975995566155e-03, 5.177975996114967e-03, 2.768029578829259e-07, 2.785211019783633e-07, 2.768714411088305e-07, 2.783880637083008e-07, 2.776770271198034e-07, 2.776770271198034e-07, 6.686541203068684e-06, 6.648304665722560e-06, 6.408898644093747e-06, 6.377769394423158e-06, 6.803645761959919e-06, 6.803645761959919e-06, 5.370859529715914e-03, 6.190853080399831e-03, 7.051788899049990e-03, 9.580297028276874e-03, 5.320617904574539e-03, 5.320617904574539e-03, 9.658161946607284e-02, 7.605487427837700e-02, 9.922711895915949e-02, 5.321882922361412e-05, 1.764436655612411e-01, 1.764436655612411e-01, 2.419945375574301e-03, 3.045799251259390e-03, 3.047368508054607e-02, 9.958313588314331e-02, 1.326220660014481e-02, 1.326220660331446e-02, 1.003333532217406e-02, 8.907386805931282e-03, 9.271908767531711e-03, 9.598550877363395e-03, 9.432323702105444e-03, 9.432323702105444e-03, 1.173489647075933e-02, 7.721795973458927e-03, 8.481502285579801e-03, 9.488116658739625e-03, 8.968028925809283e-03, 8.968028925809283e-03, 4.952314141221618e-03, 5.288932399613143e-02, 4.462913476611360e-02, 3.378952747855791e-02, 4.081854393797923e-02, 4.081854393797928e-02, 1.106992916608875e-02, 3.433492281436434e-02, 5.227627432482734e-02, 5.074970317966460e-02, 1.699151082161711e-01, 1.699151082161733e-01, 8.705504972796449e-03, 3.302434124699812e-03, 4.155207123799477e-03, 1.564102255970290e-01, 1.651468440300244e-02, 1.651468441584478e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05