
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_rpbe_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.098637570464259e+01, -2.098640097440213e+01, -2.098658473595204e+01, -2.098620730788352e+01, -2.098639491281779e+01, -2.098639491281779e+01, -3.485627092066703e+00, -3.485584575270906e+00, -3.484687531839578e+00, -3.486909627059415e+00, -3.485660139608553e+00, -3.485660139608553e+00, -7.033688345693855e-01, -7.035076923463586e-01, -7.095610476098887e-01, -7.141773077798191e-01, -7.121808258960223e-01, -7.121808258960223e-01, -2.282256546522438e-01, -2.285215342311394e-01, -8.106836770451348e-01, -2.003475627410618e-01, -2.112578059961811e-01, -2.112578059961811e-01, -1.009364494008897e-02, -1.062397932832220e-02, -5.861359302438600e-02, -5.829395465871759e-03, -7.325081590875081e-03, -7.325081590875081e-03, -5.040355200647080e+00, -5.039656206733636e+00, -5.040334168602716e+00, -5.039716911554875e+00, -5.039994605044679e+00, -5.039994605044679e+00, -2.146759195543940e+00, -2.155593983239977e+00, -2.150866099007103e+00, -2.158615885859095e+00, -2.149699477308178e+00, -2.149699477308178e+00, -5.822782827045488e-01, -6.029562804706220e-01, -5.434810641580384e-01, -5.375566858974017e-01, -5.877431071027266e-01, -5.877431071027266e-01, -1.515810955227501e-01, -2.513277159149883e-01, -1.407279091300126e-01, -1.813978596515101e+00, -1.715856292559676e-01, -1.715856292559676e-01, -4.497766541329658e-03, -5.698749801072900e-03, -4.357538022585594e-03, -9.509043572406281e-02, -5.249482297479785e-03, -5.249482297479785e-03, -5.507779072340704e-01, -5.540227993348815e-01, -5.528876459545486e-01, -5.519388371007644e-01, -5.524131339564157e-01, -5.524131339564157e-01, -5.339870509018118e-01, -5.147191167945436e-01, -5.190996930394060e-01, -5.238993811175363e-01, -5.211345489440034e-01, -5.211345489440034e-01, -6.332805539382936e-01, -2.934938134541732e-01, -3.230684854640722e-01, -3.696989663846519e-01, -3.433539679372399e-01, -3.433539679372399e-01, -4.752064103105887e-01, -5.610064986269509e-02, -7.648420839348503e-02, -3.442333208436477e-01, -1.206983168390530e-01, -1.206983168390531e-01, -1.424523243919322e-02, -1.523312330620328e-03, -3.197499979622570e-03, -1.135330819454938e-01, -4.857413173248372e-03, -4.857413173248368e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_rpbe_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.499375443913121e+01, -2.499386176120278e+01, -2.499430005603297e+01, -2.499270411659593e+01, -2.499354587875619e+01, -2.499354587875619e+01, -3.993963610558639e+00, -3.994024954953656e+00, -3.995541230869938e+00, -3.993663795368027e+00, -3.994133439953743e+00, -3.994133439953743e+00, -7.405561395699681e-01, -7.389530913755451e-01, -7.001640287424813e-01, -7.068603124862722e-01, -7.061462857383382e-01, -7.061462857383382e-01, -1.826851838822017e-01, -1.836483007245467e-01, -8.719517929366732e-01, -1.826619177752107e-01, -1.743276152743998e-01, -1.743276152743998e-01, -1.345819325345196e-02, -1.416530577109627e-02, -7.815145736254070e-02, -7.772527287829012e-03, -9.766775454500110e-03, -9.766775454500110e-03, -6.179672455500741e+00, -6.182734756601295e+00, -6.179808580316259e+00, -6.182512077122521e+00, -6.181227407800426e+00, -6.181227407800426e+00, -2.089145272929130e+00, -2.108723301028789e+00, -2.068326682823755e+00, -2.085439346660691e+00, -2.110259999030998e+00, -2.110259999030998e+00, -6.801023039517787e-01, -7.738065197346303e-01, -6.242469460083460e-01, -6.804883129015297e-01, -6.954036684455742e-01, -6.954036684455742e-01, -1.887778278103032e-01, -2.103178798415054e-01, -1.796090295970247e-01, -2.332938718739111e+00, -1.804937598465992e-01, -1.804937598465992e-01, -5.997022055106211e-03, -7.598333068097201e-03, -5.810050696780792e-03, -1.266703303330409e-01, -6.999309729973047e-03, -6.999309729973047e-03, -7.247151613865714e-01, -7.122874034557748e-01, -7.166228644235774e-01, -7.202354396068985e-01, -7.184252917112923e-01, -7.184252917112923e-01, -7.075218055271697e-01, -5.450441508211501e-01, -5.897927277124673e-01, -6.364198162218406e-01, -6.126535124078034e-01, -6.126535124078034e-01, -8.097665335044746e-01, -2.345453975570373e-01, -2.699045328758451e-01, -3.795361941376518e-01, -3.161902807724551e-01, -3.161902807724551e-01, -5.006270253946521e-01, -7.480086648352728e-02, -1.019788223428457e-01, -3.765437400719545e-01, -1.558229458979868e-01, -1.558229458979868e-01, -1.899364325225763e-02, -2.031083107493770e-03, -4.263333306163427e-03, -1.487655596487139e-01, -6.476550897664496e-03, -6.476550897664490e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_rpbe_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.565343959035560e-09, -7.565276596247893e-09, -7.564923759395258e-09, -7.565927097661553e-09, -7.565408940347317e-09, -7.565408940347317e-09, -1.045930686396108e-05, -1.045944058346203e-05, -1.046150357744599e-05, -1.044997298364645e-05, -1.045845870581109e-05, -1.045845870581109e-05, -7.036135503134083e-03, -7.051833115516973e-03, -7.376271147981261e-03, -7.159934850074472e-03, -7.224228375579304e-03, -7.224228375579304e-03, -9.018772327160818e-01, -8.920827949401615e-01, -3.879713079588462e-03, -1.145960913703138e+00, -1.165804425859014e+00, -1.165804425859014e+00, -1.290179546569406e-309, -2.760081542601534e-271, -1.983108149718766e-09, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.187602611212545e-06, -2.186962566991546e-06, -2.187562540277643e-06, -2.186997672450348e-06, -2.187284088229534e-06, -2.187284088229534e-06, -8.958964322401138e-05, -8.755205437540144e-05, -9.025240554046844e-05, -8.844491613430918e-05, -8.813028398251777e-05, -8.813028398251777e-05, -1.309784452743649e-02, -1.004475823591327e-02, -1.764078859592349e-02, -1.619749485804872e-02, -1.240473556907484e-02, -1.240473556907484e-02, -4.490630349762457e-01, -5.664703826533225e-01, -3.500269929601801e-01, -1.222637431665996e-04, -1.253244236964766e+00, -1.253244236964766e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.859273933001470e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.394367539896135e-02, -1.405759096638210e-02, -1.401759959493307e-02, -1.398485274023353e-02, -1.400129330200642e-02, -1.400129330200642e-02, -1.563206704487997e-02, -2.435614376443222e-02, -2.149901203781980e-02, -1.897443833797068e-02, -2.023997506270240e-02, -2.023997506270240e-02, -8.295582215611719e-03, -3.305999922258336e-01, -2.119393271065963e-01, -9.520105056050693e-02, -1.467520307825272e-01, -1.467520307825273e-01, -3.374498038375127e-02, -4.304006067400213e-11, -3.213588002748769e-05, -1.167821653313647e-01, -4.465514998463079e-01, -4.465514998463097e-01, -3.394801558618238e-171, 0.000000000000000e+00, 0.000000000000000e+00, -2.714856762295226e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05