
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vt84f_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.909417204910437e+03, 2.909400963678137e+03, 2.909385855068526e+03, 2.909626320854853e+03, 2.909492200358553e+03, 2.909492200358553e+03, 8.803359576771483e+01, 8.802616093523086e+01, 8.785857923797690e+01, 8.818337751878127e+01, 8.802892561258872e+01, 8.802892561258872e+01, 4.135037700837679e+00, 4.148728803200622e+00, 4.493769808868864e+00, 4.540664997262900e+00, 4.508367147927288e+00, 4.508367147927288e+00, 6.599425802432763e-01, 6.407653902451194e-01, 5.338056683088273e+00, 7.794563707665673e-01, 6.995759521685166e-01, 6.995759521685166e-01, 7.296024715785263e-01, 7.097016812785768e-01, 9.182566769856003e-01, 7.660721458424538e-01, 6.794249130319839e-01, 6.794249130319832e-01, 1.552971056185523e+02, 1.549869142129060e+02, 1.552847598513024e+02, 1.550108774446394e+02, 1.551386832756675e+02, 1.551386832756675e+02, 4.161114345439563e+01, 4.177450293702985e+01, 4.217752811651068e+01, 4.232242198023510e+01, 4.142682171235545e+01, 4.142682171235545e+01, 2.351193299444404e+00, 1.982699039147946e+00, 2.128896680939798e+00, 1.624605084625163e+00, 2.320506129611949e+00, 2.320506129611949e+00, 9.245139301165156e-01, 1.027962593131520e+00, 9.016010574049004e-01, 1.787037994023288e+01, 7.465088872324444e-01, 7.465088872324444e-01, 6.674701710122112e-01, 7.061372805600435e-01, 2.817185916449355e-01, 8.111484059661158e-01, 4.129761115202348e-01, 4.129761115202352e-01, 1.597840717308564e+00, 1.668021436801120e+00, 1.637616412919232e+00, 1.616807575050409e+00, 1.626664311662178e+00, 1.626664311662178e+00, 1.498430333933894e+00, 2.198378913967832e+00, 1.987973755538990e+00, 1.720316009435260e+00, 1.853800052773020e+00, 1.853800052773020e+00, 2.203699041865166e+00, 1.134438005974556e+00, 1.129155650890655e+00, 1.175682415915746e+00, 1.114889496081580e+00, 1.114889496081580e+00, 1.886070409899054e+00, 9.656712941861867e-01, 9.422800607064654e-01, 9.378483616801458e-01, 7.156609506656080e-01, 7.156609506656081e-01, 8.118650615792227e-01, 3.188777888300567e-01, 4.583026040831580e-01, 7.265581274218844e-01, 3.674669795903854e-01, 3.674669795903851e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vt84f_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.931868218730194e+03, 1.931910151236086e+03, 1.932049927703776e+03, 1.931427128569920e+03, 1.931760043845012e+03, 1.931760043845012e+03, 4.466042706729298e+01, 4.466258666844025e+01, 4.471575349759971e+01, 4.464156916354417e+01, 4.466517009832478e+01, 4.466517009832478e+01, 1.952106168254989e+00, 1.964225863896321e+00, 2.128995928715470e+00, 2.164413387066627e+00, 2.155874013651435e+00, 2.155874013651435e+00, -4.336767276727269e-01, -3.952337473789103e-01, 2.460984812017635e+00, -7.090265589718142e-01, -5.785323602809516e-01, -5.785323602809516e-01, -7.296021407491039e-01, -7.097012638603840e-01, -9.179577575983823e-01, -7.660721107896555e-01, -6.794248144933127e-01, -6.794248144933114e-01, 1.364680413981694e+02, 1.370771737646005e+02, 1.364935651428117e+02, 1.370313019645791e+02, 1.367777719096867e+02, 1.367777719096867e+02, 1.892626405088179e+01, 1.934531341086740e+01, 1.815758477282137e+01, 1.866032206240272e+01, 1.937964039076914e+01, 1.937964039076914e+01, 1.336664173398223e+00, 2.711431676415698e+00, 1.093035515826784e+00, 1.966486680512708e+00, 1.456728377186409e+00, 1.456728377186409e+00, -9.107529249991457e-01, -8.696727103852803e-01, -8.912717122081784e-01, 2.486715245720467e+01, -7.140358639860833e-01, -7.140358639860833e-01, -6.674701567543598e-01, -7.061372458282256e-01, -2.817185618839346e-01, -8.088027242771744e-01, -4.129760687600798e-01, -4.129760687600803e-01, 2.572599976451070e+00, 2.314920162358872e+00, 2.417109479697569e+00, 2.493732124885111e+00, 2.456541849939579e+00, 2.456541849939579e+00, 2.471494456599618e+00, 1.030604175801169e+00, 9.724108976768411e-01, 1.377528540578103e+00, 1.105560123380059e+00, 1.105560123380059e+00, 2.922068204202888e+00, -7.926036315829638e-01, -4.030838395236616e-01, 5.695385818481352e-01, 3.043048421353270e-01, 3.043048421353270e-01, 8.897480490215784e-01, -9.654327544945787e-01, -9.414354074552972e-01, 4.290694341878709e-01, -7.086629994436742e-01, -7.086629994436736e-01, -8.118638820934201e-01, -3.188777884373845e-01, -4.583025987793453e-01, -7.212007379538820e-01, -3.674669443613978e-01, -3.674669443613973e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vt84f_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.385787349172870e-06, 7.385728877790882e-06, 7.385444509690539e-06, 7.386314684735430e-06, 7.385862317076632e-06, 7.385862317076632e-06, 1.633414252640798e-03, 1.633497277256279e-03, 1.635284530071108e-03, 1.631233154217098e-03, 1.633395663955806e-03, 1.633395663955806e-03, 1.761851270604398e-01, 1.753709573545926e-01, 1.607913668119843e-01, 1.576670157383405e-01, 1.590117265875025e-01, 1.590117265875025e-01, 1.137272522373072e+01, 1.078319453831568e+01, 1.194917319230616e-01, 2.724372656597359e+01, 1.894501223542088e+01, 1.894501223542088e+01, 2.874952521326742e+05, 2.465545865582658e+05, 1.468091894514258e+03, 1.492465238426149e+06, 7.522070051643559e+05, 7.522070051643559e+05, 4.949621159781437e-04, 4.939113275891398e-04, 4.949168118831682e-04, 4.939894538111385e-04, 4.944306224424936e-04, 4.944306224424936e-04, 5.842748801789213e-03, 5.751248710463488e-03, 5.885725192266021e-03, 5.788032495772363e-03, 5.789631601398705e-03, 5.789631601398705e-03, 3.512928607597626e-01, 1.976829190173968e-01, 4.314017573357587e-01, 3.311473993363548e-01, 3.388534954778341e-01, 3.388534954778341e-01, 8.258833097483117e+01, 1.172538696605952e+01, 1.043752631420795e+02, 7.014611861290108e-03, 5.082362905552858e+01, 5.082362905552858e+01, 3.249256936990708e+06, 1.597482137302167e+06, 3.573149770352616e+06, 3.436073475375726e+02, 2.043738611025011e+06, 2.043738611025011e+06, 1.306478486223812e-01, 2.475762201604596e-01, 2.128828386513451e-01, 1.791885972225644e-01, 1.966419355737574e-01, 1.966419355737574e-01, 9.072221417738255e-02, 4.540843700315417e-01, 4.917576967826213e-01, 4.550688783928061e-01, 4.885876803332472e-01, 4.885876803332472e-01, 1.799613867881345e-01, 5.658322820596297e+00, 3.010712375719890e+00, 1.166913083805053e+00, 1.610209323796742e+00, 1.610209323796741e+00, 5.718926224914135e-01, 1.674368296766356e+03, 6.606727536127133e+02, 1.606524564603059e+00, 1.662222338128409e+02, 1.662222338128409e+02, 1.022741895002582e+05, 8.363882289937356e+07, 9.043637223563571e+06, 2.008305458561850e+02, 2.579643181763897e+06, 2.579643181763904e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05