
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_wc_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.084480773799212e+01, -2.084484004631537e+01, -2.084504305203079e+01, -2.084456114058621e+01, -2.084480521128259e+01, -2.084480521128259e+01, -3.445209054235243e+00, -3.445180414002136e+00, -3.444602536686262e+00, -3.446261261141002e+00, -3.445257764977171e+00, -3.445257764977171e+00, -6.853590235414626e-01, -6.851779524384745e-01, -6.819204917088996e-01, -6.868354408800974e-01, -6.851952877411479e-01, -6.851952877411479e-01, -2.042256275406375e-01, -2.053127764586624e-01, -7.931068545436887e-01, -1.755924805571849e-01, -1.857888529548379e-01, -1.857888529548379e-01, -1.008299808023360e-02, -1.061126635064672e-02, -5.740749649352397e-02, -5.827399407367881e-03, -7.320653746818632e-03, -7.320653746818632e-03, -5.020898851996446e+00, -5.020480135227161e+00, -5.020889406906025e+00, -5.020519631946333e+00, -5.020681331101122e+00, -5.020681331101122e+00, -2.056436347411546e+00, -2.067453287430530e+00, -2.054444142168451e+00, -2.064179454963263e+00, -2.063475623397293e+00, -2.063475623397293e+00, -5.770065746865152e-01, -6.024441727796818e-01, -5.373608725687505e-01, -5.367290004326080e-01, -5.833405829069048e-01, -5.833405829069048e-01, -1.376428791674670e-01, -2.207004101521297e-01, -1.288233825616299e-01, -1.812600976788076e+00, -1.515909075247082e-01, -1.515909075247082e-01, -4.496710301154653e-03, -5.696727532225102e-03, -4.355283833197488e-03, -9.035599747027340e-02, -5.246793382884695e-03, -5.246793382884695e-03, -5.507198580164968e-01, -5.535944913461298e-01, -5.526268004049617e-01, -5.517864579584154e-01, -5.522100071378654e-01, -5.522100071378654e-01, -5.339742379804783e-01, -5.020939530775541e-01, -5.124558861375174e-01, -5.214364150750047e-01, -5.168344708958982e-01, -5.168344708958982e-01, -6.326382731277264e-01, -2.614899025230271e-01, -2.959380317238055e-01, -3.584018454294973e-01, -3.243256010560693e-01, -3.243256010560693e-01, -4.630920117277482e-01, -5.506965669294386e-02, -7.411310406785247e-02, -3.377758274011696e-01, -1.110295488649465e-01, -1.110295488649465e-01, -1.421889370030923e-02, -1.523225841529530e-03, -3.196946033936698e-03, -1.053301436809432e-01, -4.855017579384378e-03, -4.855017579384373e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_wc_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.547284886460890e+01, -2.547293469597932e+01, -2.547331448191466e+01, -2.547203754315554e+01, -2.547270687773718e+01, -2.547270687773718e+01, -4.122456055924848e+00, -4.122479193277868e+00, -4.123116238606406e+00, -4.122792937761397e+00, -4.122583173464141e+00, -4.122583173464141e+00, -7.896529022563176e-01, -7.886858152424117e-01, -7.654912894267358e-01, -7.719601272246441e-01, -7.706753339267823e-01, -7.706753339267823e-01, -1.941494236513746e-01, -1.978718607694396e-01, -9.218983666106241e-01, -1.573261751064831e-01, -1.657463834448736e-01, -1.657463834448736e-01, -1.341641964367207e-02, -1.411552682734011e-02, -7.391792171136655e-02, -7.764604248719035e-03, -9.749280742423169e-03, -9.749280742423169e-03, -6.248536516916696e+00, -6.250672109110804e+00, -6.248634366613284e+00, -6.250519610087385e+00, -6.249618585029455e+00, -6.249618585029455e+00, -2.295311512860958e+00, -2.312608106452235e+00, -2.281583387979249e+00, -2.296923659680302e+00, -2.311453958742040e+00, -2.311453958742040e+00, -6.973974958467996e-01, -7.757460460487244e-01, -6.437819878923234e-01, -6.835609976329838e-01, -7.101492347909775e-01, -7.101492347909775e-01, -1.490662723901774e-01, -1.960636885422321e-01, -1.431906214758205e-01, -2.338172356852394e+00, -1.450017525823453e-01, -1.450017525823453e-01, -5.992820709164379e-03, -7.590307859317453e-03, -5.801131085633433e-03, -1.101018783648785e-01, -6.988669019936460e-03, -6.988669019936460e-03, -7.249430426653691e-01, -7.139137857458945e-01, -7.176250320556418e-01, -7.208265762144754e-01, -7.192094939096388e-01, -7.192094939096388e-01, -7.075726036119500e-01, -5.798547475315472e-01, -6.106156497862489e-01, -6.450153906863425e-01, -6.269093987639405e-01, -6.269093987639405e-01, -8.121862241862752e-01, -2.447452232757314e-01, -3.018343157571606e-01, -4.087657189758672e-01, -3.531360714682655e-01, -3.531360714682655e-01, -5.336912886649167e-01, -7.116560981668582e-02, -9.372582666571799e-02, -3.954775742827440e-01, -1.251968009981146e-01, -1.251968009981146e-01, -1.889139539598431e-02, -2.030737722915481e-03, -4.261127097640499e-03, -1.215269038478288e-01, -6.467067801240247e-03, -6.467067801240241e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_wc_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.874449400053272e-09, -5.874433600317301e-09, -5.874263355046122e-09, -5.874500482522152e-09, -5.874390452249799e-09, -5.874390452249799e-09, -7.540428356232189e-06, -7.540905242200215e-06, -7.551202083061212e-06, -7.527595406977545e-06, -7.540273038918943e-06, -7.540273038918943e-06, -4.428473684445331e-03, -4.424189821671242e-03, -4.311350435993476e-03, -4.196599167493285e-03, -4.241395432030282e-03, -4.241395432030282e-03, -5.795549960599203e-01, -5.592031821842652e-01, -2.517238360612036e-03, -1.041889698769418e+00, -8.902086936944942e-01, -8.902086936944941e-01, -4.075072002448390e+00, -4.276781862836136e+00, -1.574236808474970e+00, -3.844022748292465e+00, -4.812215924369061e+00, -4.812215924369072e+00, -1.804102043888259e-06, -1.806019377918051e-06, -1.804169930939058e-06, -1.805862813082378e-06, -1.805084792351633e-06, -1.805084792351633e-06, -5.174733845561163e-05, -5.078741061837673e-05, -5.166551786098645e-05, -5.080053075140878e-05, -5.127450215988792e-05, -5.127450215988792e-05, -9.788423923445330e-03, -9.170672524863371e-03, -1.277431665329658e-02, -1.433162742536632e-02, -9.507058335634156e-03, -9.507058335634156e-03, -1.160785571872755e+00, -4.457992376909251e-01, -1.245815252826326e+00, -1.121766090083604e-04, -1.482480036311741e+00, -1.482480036311741e+00, -5.098687528675900e+00, -4.520762242465242e+00, -2.812873323870282e+01, -1.649529093527760e+00, -1.309359549059595e+01, -1.309359549059593e+01, -1.350280949741637e-02, -1.288790587303427e-02, -1.309723909494110e-02, -1.327655174060567e-02, -1.318631704098382e-02, -1.318631704098382e-02, -1.539419379947231e-02, -1.545086894488173e-02, -1.526373673810050e-02, -1.534543133412285e-02, -1.531758975819131e-02, -1.531758975819131e-02, -7.510294714890524e-03, -2.191098735817916e-01, -1.222063618051243e-01, -5.801554664431689e-02, -8.217456495626478e-02, -8.217456495626480e-02, -2.125587850614824e-02, -1.469997924954507e+00, -1.339188857061874e+00, -7.776153357629396e-02, -1.996889227269468e+00, -1.996889227269470e+00, -3.171227183275400e+00, -2.262908306257918e+01, -1.086010451456578e+01, -1.965894649875770e+00, -1.655587782076208e+01, -1.655587782076212e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05