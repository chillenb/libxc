
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_17_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.467145265432534e+01, -2.467151561726416e+01, -2.467195301499790e+01, -2.467092769786412e+01, -2.467148443648479e+01, -2.467148443648479e+01, -3.410540794720950e+00, -3.410663377454587e+00, -3.414633479573573e+00, -3.415626368051886e+00, -3.410576555065019e+00, -3.410576555065019e+00, -5.923532301579931e-01, -5.917837890008053e-01, -5.780482993273887e-01, -5.879631480780606e-01, -5.921522007646073e-01, -5.921522007646073e-01, -1.789883681945168e-01, -1.817042908976436e-01, -6.725496758465351e-01, -1.162034088485677e-01, -1.797337189345873e-01, -1.797337189345873e-01, -8.849560268537828e-03, -9.267501323032374e-03, -3.764336846135908e-02, -4.202289293910116e-03, -9.172470094662741e-03, -9.172470094662741e-03, -6.033423262331225e+00, -6.033828578348370e+00, -6.033513774389661e+00, -6.033826239227302e+00, -6.033566117263788e+00, -6.033566117263788e+00, -2.091170073969361e+00, -2.123641100981389e+00, -2.087779858059397e+00, -2.113755688659386e+00, -2.117307964092524e+00, -2.117307964092524e+00, -6.154453677175336e-01, -6.636902561826373e-01, -5.578023594359949e-01, -5.799269962897200e-01, -6.521721724906504e-01, -6.521721724906504e-01, -8.519641676015551e-02, -1.725063703719954e-01, -8.418497364427112e-02, -1.916916885875371e+00, -1.001145667829799e-01, -1.001145667829799e-01, -4.068503531091244e-03, -4.639270131367782e-03, -3.491913281061663e-03, -5.224093496871694e-02, -4.221096418397037e-03, -4.221096418397037e-03, -6.728218736845204e-01, -6.745126147127705e-01, -6.740058288927099e-01, -6.735357013238849e-01, -6.737754781892846e-01, -6.737754781892846e-01, -6.442033008599175e-01, -5.734322519410575e-01, -6.012284783103171e-01, -6.213360017408387e-01, -6.111400889228872e-01, -6.111400889228872e-01, -6.816939576644808e-01, -2.268928382794274e-01, -2.753814615705710e-01, -3.549984676894171e-01, -3.176236388803512e-01, -3.176236388803511e-01, -4.824092976391708e-01, -3.492980462377201e-02, -4.722416468193079e-02, -3.374905312956011e-01, -6.914244141601762e-02, -6.914244141601762e-02, -1.027029746882881e-02, -1.266928933256275e-03, -2.425954704123803e-03, -6.590681249443660e-02, -3.586554899872921e-03, -3.586554899872917e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_17_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.963626981740623e+01, -2.963636530952722e+01, -2.963671600859733e+01, -2.963514672804940e+01, -2.963632089819191e+01, -2.963632089819191e+01, -4.831169060348583e+00, -4.831307296617499e+00, -4.835628282469596e+00, -4.834515190325979e+00, -4.831229239093287e+00, -4.831229239093287e+00, -8.132222034275461e-01, -8.119673247799650e-01, -7.825521317234370e-01, -7.988410769892680e-01, -8.127794035851125e-01, -8.127794035851125e-01, -2.361746920956792e-01, -2.401410810337674e-01, -8.921750107579300e-01, -1.522100727908975e-01, -2.372980785552036e-01, -2.372980785552036e-01, -1.136881484980481e-02, -1.191308878356137e-02, -4.890565050340906e-02, -5.550082257966264e-03, -1.178626112883616e-02, -1.178626112883618e-02, -7.287521262824048e+00, -7.291514181835754e+00, -7.287891819329639e+00, -7.290998613072812e+00, -7.289616367825772e+00, -7.289616367825772e+00, -2.696782731440226e+00, -2.720158962273491e+00, -2.688312852452518e+00, -2.706834418104539e+00, -2.723898425044171e+00, -2.723898425044171e+00, -8.268279794588790e-01, -9.386455901125849e-01, -7.787241116734389e-01, -8.438448412853554e-01, -8.604983433142234e-01, -8.604983433142234e-01, -1.109804954941556e-01, -2.253042319831915e-01, -1.096579824166219e-01, -2.842366068616931e+00, -1.301132039920557e-01, -1.301132039920557e-01, -5.311032010159982e-03, -6.097420594625976e-03, -4.533710581446478e-03, -6.867927806364277e-02, -5.576275053639164e-03, -5.576275053639156e-03, -8.942254797343065e-01, -8.722610018262186e-01, -8.798689780216791e-01, -8.859560413623656e-01, -8.828914455687256e-01, -8.828914455687256e-01, -8.707568358159633e-01, -6.916924219408837e-01, -7.265579275331817e-01, -7.714775396700164e-01, -7.474364093034133e-01, -7.474364093034135e-01, -9.826528517488587e-01, -2.976884839521891e-01, -3.638146155230730e-01, -4.807968853234028e-01, -4.209675990913975e-01, -4.209675990913975e-01, -6.317981116689254e-01, -4.514335717478583e-02, -6.148327627287446e-02, -4.621950637999825e-01, -8.988148737102059e-02, -8.988148737102063e-02, -1.355026985168852e-02, -1.683009887275050e-03, -3.139575232253940e-03, -8.574580649602616e-02, -4.742453777239637e-03, -4.742453777239630e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.175894845536339e-08, -2.175886583998932e-08, -2.175843718396004e-08, -2.175981252262285e-08, -2.175890511416057e-08, -2.175890511416057e-08, -1.589149919273897e-05, -1.589486819098893e-05, -1.599480831604587e-05, -1.596263514955501e-05, -1.589205748077974e-05, -1.589205748077974e-05, -4.344851892197074e-03, -4.328058830741238e-03, -3.919066849216133e-03, -4.251804911569209e-03, -4.339438742307020e-03, -4.339438742307020e-03, -7.374227318406296e-01, -7.472155917186285e-01, -9.766008627229543e-04, -3.048622047202748e-01, -7.391755175346210e-01, -7.391755175346210e-01, -5.217367998357581e+01, -4.693907806838547e+01, -1.671825607863943e+00, -4.805415883437448e+01, -4.937922735995316e+01, -4.937922735995313e+01, -6.604573273164738e-06, -6.605750756671828e-06, -6.604789705139696e-06, -6.605698410971830e-06, -6.605056784431048e-06, -6.605056784431048e-06, -1.463385738038007e-04, -1.490384503671231e-04, -1.458340378040508e-04, -1.481423045428219e-04, -1.487756660028336e-04, -1.487756660028336e-04, -2.953089548446062e-02, -2.593657928609038e-02, -3.007626358265686e-02, -3.132561369945108e-02, -2.820266804932518e-02, -2.820266804932518e-02, -4.377591209452392e-01, -3.112512780785638e-01, -5.020782144311565e-01, -2.554409089146962e-04, -5.730549700923829e-01, -5.730549700923829e-01, -1.181874321263782e+02, -6.167168261407772e+01, -3.427411789061627e+02, -5.785931562743408e-01, -5.823421353640556e+01, -5.823421353640542e+01, -3.966460849404625e-02, -3.991396427244290e-02, -3.984599458557086e-02, -3.977767856961409e-02, -3.981334579392562e-02, -3.981334579392562e-02, -4.431638899131939e-02, -5.471974724134412e-02, -5.398373632625152e-02, -5.141046601986648e-02, -5.286197775824745e-02, -5.286197775824743e-02, -1.997565092744721e-02, -2.464599269275607e-01, -1.999096655794722e-01, -1.352476544500792e-01, -1.804104490546924e-01, -1.804104490546925e-01, -6.307395043544889e-02, -2.033700093452791e+00, -1.006962704843161e+00, -2.063071499873914e-01, -8.391682928243369e-01, -8.391682928243366e-01, -9.436759413994670e+00, -2.766203268488498e+02, -5.718825513105800e+02, -9.917226755101534e-01, -8.753429892692657e+01, -8.753429892692684e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.127686911700283e-03, 3.127674704688210e-03, 3.127654104489767e-03, 3.127855796875226e-03, 3.127680159094792e-03, 3.127680159094792e-03, 9.611543109336868e-03, 9.613963754834158e-03, 9.688501547621714e-03, 9.688405957424156e-03, 9.611979942056080e-03, 9.611979942056080e-03, 1.413017064526141e-02, 1.406344530403217e-02, 1.269248105454730e-02, 1.496714514286521e-02, 1.410918070368384e-02, 1.410918070368384e-02, 9.737740706714648e-02, 1.012680304474802e-01, 2.001035162309248e-03, 1.046101408201594e-02, 9.825275210412333e-02, 9.825275210412333e-02, 8.661703922599411e-04, 8.916655824976114e-04, 2.082059644598607e-03, 6.187484427630743e-05, 9.120819841482166e-04, 9.120819841482026e-04, 1.258097546931642e-02, 1.256680067575445e-02, 1.257969451429019e-02, 1.256866327862360e-02, 1.257349079554972e-02, 1.257349079554972e-02, 2.257750834504329e-02, 2.356268996694441e-02, 2.257144750348437e-02, 2.338292552286518e-02, 2.323540850412501e-02, 2.323540850412501e-02, 8.818413909627275e-02, 8.132770164703149e-02, 7.458624739884577e-02, 7.634668015168314e-02, 9.228316550634900e-02, 9.228316550634900e-02, 6.634296784992898e-03, 4.203643663194867e-02, 7.363092990306796e-03, 2.221399661835494e-02, 1.499382000656057e-02, 1.499382000656057e-02, 1.637422154705038e-04, 1.159797799155741e-04, 3.179537347103463e-04, 1.689804041414538e-03, 7.567947812528348e-05, 7.567947812528210e-05, 9.275517917780082e-02, 9.815895472523554e-02, 9.626255408614369e-02, 9.476347441788864e-02, 9.551610484972478e-02, 9.551610484972478e-02, 9.459926141299151e-02, 1.281336291542163e-01, 1.258374371380304e-01, 1.164990711234657e-01, 1.216254573044988e-01, 1.216254573044988e-01, 7.265698034088089e-02, 6.975874205599311e-02, 8.856586406400660e-02, 1.009843884270092e-01, 1.069337578831746e-01, 1.069337578831747e-01, 1.054989817296643e-01, 2.097162204314116e-03, 2.474475622751779e-03, 1.205316106314344e-01, 6.941770713098934e-03, 6.941770713099019e-03, 1.823116522163729e-04, 8.544202370373454e-06, 1.830233320207162e-04, 7.030603780461425e-03, 6.849159362807284e-05, 6.849159362807174e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05