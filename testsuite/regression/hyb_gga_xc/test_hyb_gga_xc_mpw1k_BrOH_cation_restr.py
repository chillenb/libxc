
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1k_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.212099219976594e+01, -1.212100820513731e+01, -1.212111801683486e+01, -1.212087909223686e+01, -1.212099878665575e+01, -1.212099878665575e+01, -2.048999675029580e+00, -2.048982214385216e+00, -2.048627867908838e+00, -2.049624949350050e+00, -2.049027075643650e+00, -2.049027075643650e+00, -4.316836461530444e-01, -4.314643672353347e-01, -4.271687670494366e-01, -4.301861174921549e-01, -4.292637858411226e-01, -4.292637858411226e-01, -1.315671604263437e-01, -1.326352976939793e-01, -4.965067015336453e-01, -1.110519286745348e-01, -1.181933286600041e-01, -1.181933286600042e-01, -4.887836535075870e-04, -5.768410705346694e-04, -3.182378467707062e-02, -1.027751817868798e-04, -2.146453635918843e-04, -2.146453635918841e-04, -2.962027190905592e+00, -2.961800477721297e+00, -2.962022340711351e+00, -2.961822104624515e+00, -2.961909110634317e+00, -2.961909110634317e+00, -1.243169289374489e+00, -1.249371755942943e+00, -1.242771933291521e+00, -1.248236832556847e+00, -1.246803625968492e+00, -1.246803625968492e+00, -3.746713877136736e-01, -4.023080844295884e-01, -3.496045773907262e-01, -3.604382799027833e-01, -3.797040794180730e-01, -3.797040794180730e-01, -8.971234012558785e-02, -1.395704226728356e-01, -8.461573330679617e-02, -1.114131523522838e+00, -9.638529448524452e-02, -9.638529448524452e-02, -5.674117284197284e-05, -1.037171260772131e-04, -1.110433956071988e-04, -6.061801409508743e-02, -1.325948420018351e-04, -1.325948420018353e-04, -3.759355306855043e-01, -3.732019015362903e-01, -3.741148686907816e-01, -3.749160803920142e-01, -3.745103803163884e-01, -3.745103803163884e-01, -3.671014203637964e-01, -3.230801325599384e-01, -3.338875956096816e-01, -3.456917205637027e-01, -3.394704581730100e-01, -3.394704581730100e-01, -4.197502762579672e-01, -1.664021982701376e-01, -1.900285871634226e-01, -2.354576843577634e-01, -2.104878519293514e-01, -2.104878519293514e-01, -2.995311405278034e-01, -2.892325720085068e-02, -4.711374347673943e-02, -2.265198341621613e-01, -7.328226001423424e-02, -7.328226001423423e-02, -1.149277866252498e-03, -5.521440216365516e-06, -3.083102665172117e-05, -7.002578981805331e-02, -1.186001282006815e-04, -1.186001282006781e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1k_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.455711659441708e+01, -1.455717123461011e+01, -1.455740348078142e+01, -1.455659077214098e+01, -1.455701812825217e+01, -1.455701812825217e+01, -2.432814154321405e+00, -2.432840396180316e+00, -2.433504744802908e+00, -2.432793499254728e+00, -2.432901872259111e+00, -2.432901872259111e+00, -5.178699224841115e-01, -5.169602306919654e-01, -4.943541500863144e-01, -4.984795766500443e-01, -4.980138141595412e-01, -4.980138141595412e-01, -1.404248594462437e-01, -1.437695891972182e-01, -5.973439441192312e-01, -1.016840062060506e-01, -1.155450787831401e-01, -1.155450787831401e-01, -1.799173694405834e-03, -2.121604991921563e-03, -7.196226050619395e-02, -3.781305392202870e-04, -7.908043591790462e-04, -7.908043591790509e-04, -3.676630166068047e+00, -3.678208658418210e+00, -3.676700872869460e+00, -3.678094370656366e+00, -3.677431057534819e+00, -3.677431057534819e+00, -1.346732593671629e+00, -1.357676570953316e+00, -1.336185631084849e+00, -1.345811735739632e+00, -1.357945844652545e+00, -1.357945844652545e+00, -4.763880464523108e-01, -5.249358477206665e-01, -4.429384638887051e-01, -4.697713487260354e-01, -4.849873071092958e-01, -4.849873071092958e-01, -8.347152868367549e-02, -1.322774000959775e-01, -8.126343143909068e-02, -1.448550099632602e+00, -8.568565757064689e-02, -8.568565757064689e-02, -2.084769529512520e-04, -3.816412399039772e-04, -4.090144475964980e-04, -7.681926236712369e-02, -4.883868699257366e-04, -4.883868699257374e-04, -4.892950970637537e-01, -4.870138435388971e-01, -4.881705636072342e-01, -4.888601820896137e-01, -4.885523525594058e-01, -4.885523525594058e-01, -4.770637062982455e-01, -3.967040333675498e-01, -4.221634920889757e-01, -4.468557118010122e-01, -4.344934269951696e-01, -4.344934269951696e-01, -5.474209323197077e-01, -1.694648518128622e-01, -2.102288579084923e-01, -2.922874232555147e-01, -2.492932917805842e-01, -2.492932917805842e-01, -3.689668215424876e-01, -6.956425915034051e-02, -8.211503949688866e-02, -2.891740323607295e-01, -7.211641528454674e-02, -7.211641528454671e-02, -4.202704041127819e-03, -2.018302044283408e-05, -1.131458319255815e-04, -7.219041115682340e-02, -4.368056773836015e-04, -4.368056773835949e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1k_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.009394557124714e-09, -4.009372739851841e-09, -4.009226040441467e-09, -4.009551664652768e-09, -4.009388108272743e-09, -4.009388108272743e-09, -4.554194465098330e-06, -4.554260783181285e-06, -4.555360449889980e-06, -4.550143180541959e-06, -4.553855440838618e-06, -4.553855440838618e-06, -1.756200352091026e-03, -1.768190438076283e-03, -2.045749327154460e-03, -1.984948777121955e-03, -1.995088420911933e-03, -1.995088420911933e-03, -2.456017482830289e-01, -2.287551629074923e-01, -1.027823214282987e-03, -6.209457821866028e-01, -4.450347515411213e-01, -4.450347515411213e-01, 1.695564700338980e+02, 1.761979808796312e+02, 1.770817285651322e+01, 1.761398588837290e+02, 2.094998231683570e+02, 2.094998231683597e+02, -1.023101436917815e-06, -1.022533746319669e-06, -1.023072217205668e-06, -1.022571328107789e-06, -1.022817294302765e-06, -1.022817294302765e-06, -3.509072546466435e-05, -3.431928119977472e-05, -3.535449880979466e-05, -3.466746878594979e-05, -3.453391330020553e-05, -3.453391330020553e-05, -1.707671094033303e-03, 6.392105995638624e-04, -2.328310397672541e-03, 7.052876113348566e-04, -1.444657853434803e-03, -1.444657853434803e-03, -1.213657552395934e+00, -2.396707062335736e-01, -1.371963700244567e+00, -2.414647419103070e-05, -1.103336233914771e+00, -1.103336233914771e+00, 2.424675980618855e+02, 2.064489494431013e+02, 1.241181738462293e+03, -6.335040204512213e-01, 5.782565433692743e+02, 5.782565433692727e+02, 1.988040799397879e-03, 1.246077803749026e-03, 1.665044376250572e-03, 1.945640677852307e-03, 1.821380602265530e-03, 1.821380602265530e-03, 2.803188180177494e-03, -4.507452926796739e-03, -2.832880330892455e-03, -8.679226824733714e-04, -1.909889557161868e-03, -1.909889557161868e-03, 3.212976771979148e-04, -1.066427089154189e-01, -5.275341263471112e-02, -1.265473339278523e-02, -2.761953125348129e-02, -2.761953125348132e-02, -5.778002911151011e-03, 2.015923658146033e+01, 5.076970655622312e+00, -7.605334522619904e-03, -2.230932046884678e+00, -2.230932046884676e+00, 1.261478138299603e+02, 1.261072952277865e+03, 5.330678981325354e+02, -2.195493700726632e+00, 7.336118171833294e+02, 7.336118171833178e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05