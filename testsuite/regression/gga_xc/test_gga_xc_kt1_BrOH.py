
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_kt1_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.214448729142619e+01, -2.214448496377195e+01, -2.214459454946482e+01, -2.214463029723481e+01, -2.214448506104957e+01, -2.214448506104957e+01, -3.823976817508035e+00, -3.823904849770817e+00, -3.822083091309949e+00, -3.826018038739846e+00, -3.823955744244483e+00, -3.823955744244483e+00, -7.978056357106837e-01, -7.981821731574800e-01, -8.121124736056767e-01, -8.174949014643378e-01, -7.979363083052768e-01, -7.979363083052768e-01, -2.094622153267717e-01, -2.119667859539340e-01, -9.288669919904450e-01, -1.563982001894403e-01, -2.102126909277410e-01, -2.102126909277410e-01, -1.530036557562211e-02, -1.597229911358816e-02, -5.730998683132244e-02, -7.690234989350562e-03, -1.581800551965639e-02, -1.581800551965639e-02, -5.335234633214279e+00, -5.333198352257075e+00, -5.335036072374464e+00, -5.333452547267558e+00, -5.334175799081751e+00, -5.334175799081751e+00, -2.579012197120456e+00, -2.581697064105433e+00, -2.593368348806890e+00, -2.595221859786620e+00, -2.562863858535335e+00, -2.562863858535335e+00, -6.387794299703693e-01, -6.624749497313506e-01, -6.068977103036653e-01, -6.106357008697931e-01, -6.591916998559233e-01, -6.591916998559233e-01, -1.183686173482649e-01, -2.120154533721013e-01, -1.169721221435824e-01, -1.932066437574080e+00, -1.349071954871543e-01, -1.349071954871543e-01, -7.433178349290962e-03, -8.432241499400684e-03, -6.418663760612413e-03, -7.723578398366059e-02, -7.723150841068013e-03, -7.723150841068013e-03, -6.232474372034642e-01, -6.260053588701930e-01, -6.250258060089812e-01, -6.242591412017706e-01, -6.246426817701862e-01, -6.246426817701862e-01, -6.034760861918728e-01, -5.670671249883009e-01, -5.792259221589664e-01, -5.890461172352790e-01, -5.840263010965203e-01, -5.840263010965203e-01, -6.946555886788115e-01, -2.633000705991803e-01, -3.097991744573429e-01, -3.891384534936313e-01, -3.473296074237557e-01, -3.473296074237557e-01, -5.084929902478352e-01, -5.349127402125789e-02, -7.023059591268577e-02, -3.647925413312739e-01, -9.804148977285737e-02, -9.804148977285737e-02, -1.772162603819004e-02, -2.442786883666834e-03, -4.530662331192576e-03, -9.396816192972922e-02, -6.620529149731377e-03, -6.620529149731370e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_kt1_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.372672217821237e+01, -2.372685375724095e+01, -2.372737541778877e+01, -2.372522202976207e+01, -2.372679211475705e+01, -2.372679211475705e+01, -3.776533595795559e+00, -3.776647677233504e+00, -3.779931813400064e+00, -3.775049716063837e+00, -3.776607953258717e+00, -3.776607953258717e+00, -8.013771805809533e-01, -7.997289049532037e-01, -7.544865733946184e-01, -7.592206556924543e-01, -8.007815536720286e-01, -8.007815536720286e-01, -2.707523751234108e-01, -2.740294114685647e-01, -9.448458997717653e-01, -2.023222292849504e-01, -2.717356736162145e-01, -2.717356736162145e-01, -2.003078221434546e-02, -2.090430820656092e-02, -7.443437276001280e-02, -1.011322331522869e-02, -2.070375398685190e-02, -2.070375398685190e-02, -6.062137398035555e+00, -6.067154574525410e+00, -6.062645042971204e+00, -6.066546342610235e+00, -6.064719270698689e+00, -6.064719270698689e+00, -1.636520615606417e+00, -1.667219597512537e+00, -1.603667375524084e+00, -1.627941022222301e+00, -1.699025553375443e+00, -1.699025553375443e+00, -7.642711043412316e-01, -8.444921561743362e-01, -7.307781294723900e-01, -7.725955644321689e-01, -7.918964364138669e-01, -7.918964364138669e-01, -1.532840786887261e-01, -2.724981270937444e-01, -1.515184578947348e-01, -2.399549083683834e+00, -1.746706122663672e-01, -1.746706122663672e-01, -9.777188855457839e-03, -1.108272755172735e-02, -8.450072854127287e-03, -1.002058218059335e-01, -1.015625487670522e-02, -1.015625487670522e-02, -8.114127987186786e-01, -8.030139954751774e-01, -8.060086388958090e-01, -8.083342851767692e-01, -8.071696448556103e-01, -8.071696448556103e-01, -7.877464875208202e-01, -6.697554011817816e-01, -7.018308639778509e-01, -7.334165120194373e-01, -7.172016879709194e-01, -7.172016879709194e-01, -8.805319524143383e-01, -3.357768338223904e-01, -3.928358865771947e-01, -4.915996296472068e-01, -4.394824086204047e-01, -4.394824086204047e-01, -6.152371923650644e-01, -6.949244806412114e-02, -9.113915270149689e-02, -4.674279169491598e-01, -1.271156350968115e-01, -1.271156350968115e-01, -2.317721574073878e-02, -3.230998150007618e-03, -5.975855230893354e-03, -1.218665306498317e-01, -8.714260121262953e-03, -8.714260121262942e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_kt1_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.457024970462253e-08, -1.457010823848453e-08, -1.456937297014644e-08, -1.457169168258781e-08, -1.457017597084265e-08, -1.457017597084265e-08, -2.072492522972928e-05, -2.072475136940251e-05, -2.071656443704143e-05, -2.071311739184369e-05, -2.072448291414267e-05, -2.072448291414267e-05, -1.015914158946932e-02, -1.018676167645068e-02, -1.092524028455702e-02, -1.070827922928989e-02, -1.016920785474571e-02, -1.016920785474571e-02, -2.967538912784231e-02, -2.965795297400472e-02, -5.964807461309258e-03, -2.990960843420687e-02, -2.967022933565853e-02, -2.967022933565853e-02, -2.999999680698964e-02, -2.999999615639031e-02, -2.999895098535874e-02, -2.999999983087901e-02, -2.999999631412034e-02, -2.999999631412034e-02, -4.133105599519598e-06, -4.130166998725898e-06, -4.132795429805716e-06, -4.130510627030078e-06, -4.131611680933994e-06, -4.131611680933994e-06, -2.062188459082051e-04, -2.005460883079356e-04, -2.085686676257886e-04, -2.040614854708459e-04, -1.994333330279873e-04, -1.994333330279873e-04, -1.413736789048140e-02, -1.170886799054341e-02, -1.570943093078594e-02, -1.444639867579984e-02, -1.301702198290690e-02, -1.301702198290690e-02, -2.997360524767782e-02, -2.966609066098439e-02, -2.997493157383263e-02, -2.226988248776979e-04, -2.995284181123701e-02, -2.995284181123701e-02, -2.999999985353185e-02, -2.999999975014974e-02, -2.999999992115428e-02, -2.999603894951902e-02, -2.999999982779374e-02, -2.999999982779374e-02, -1.320377687432580e-02, -1.336225912943033e-02, -1.330530434659269e-02, -1.326147249653267e-02, -1.328339854622248e-02, -1.328339854622248e-02, -1.417455691278754e-02, -1.837910076203098e-02, -1.711641315303685e-02, -1.596571141489470e-02, -1.654876906531310e-02, -1.654876906531310e-02, -1.038787817485716e-02, -2.917814223383646e-02, -2.840796328686458e-02, -2.609364818287020e-02, -2.748232321313896e-02, -2.748232321313897e-02, -2.115017212748282e-02, -2.999922810030482e-02, -2.999740754995131e-02, -2.680227909122726e-02, -2.998853887524246e-02, -2.998853887524247e-02, -2.999999397580472e-02, -2.999999999859621e-02, -2.999999998171961e-02, -2.999050515255835e-02, -2.999999991016322e-02, -2.999999991016322e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05