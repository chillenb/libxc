
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mspbel_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.215066009037297e+01, -2.215070766960534e+01, -2.215096701599428e+01, -2.215025848941545e+01, -2.215062293256741e+01, -2.215062293256741e+01, -3.377236039492448e+00, -3.377238707940996e+00, -3.377520050050688e+00, -3.378960643149769e+00, -3.377950513002200e+00, -3.377950513002200e+00, -6.498355334051437e-01, -6.495410501291206e-01, -6.448840400887863e-01, -6.537071065966189e-01, -6.541070058947790e-01, -6.541070058947790e-01, -2.039019466081150e-01, -2.048311142377778e-01, -7.261882286556912e-01, -1.668984724588317e-01, -1.960625183695283e-01, -1.960625183695283e-01, -1.010108097925851e-02, -1.063566561805193e-02, -5.706710119167110e-02, -5.828194691586444e-03, -8.131825392133848e-03, -8.131825392133848e-03, -5.367413340357342e+00, -5.367681359932888e+00, -5.367433023804716e+00, -5.367669498636094e+00, -5.367545129136892e+00, -5.367545129136892e+00, -2.109405830234995e+00, -2.127564323389655e+00, -2.109520756268978e+00, -2.125248594768972e+00, -2.119645655444893e+00, -2.119645655444893e+00, -5.961298815095417e-01, -6.352894709998788e-01, -5.337685627879087e-01, -5.378415698983687e-01, -6.068736316549577e-01, -6.068736316549579e-01, -1.311763645214862e-01, -2.144102421963532e-01, -1.232649612220366e-01, -1.830936489723039e+00, -1.443860568393064e-01, -1.443860568393064e-01, -4.500023878776848e-03, -5.697343613399609e-03, -4.362356892264690e-03, -8.801805414856376e-02, -5.479511911928404e-03, -5.479511911928405e-03, -5.987366538181845e-01, -5.981412162075135e-01, -5.983536491906319e-01, -5.985223866737119e-01, -5.984374016085111e-01, -5.984374016085111e-01, -5.809945097937992e-01, -5.241181858754087e-01, -5.398362380144384e-01, -5.556501042116231e-01, -5.473562302389819e-01, -5.473562302389819e-01, -6.578643310951474e-01, -2.614540671970265e-01, -2.958696445445252e-01, -3.536436942178139e-01, -3.242503108626231e-01, -3.242503108626232e-01, -4.738832244307966e-01, -5.473948687192605e-02, -7.312553094972488e-02, -3.398199557046835e-01, -1.064609509486607e-01, -1.064609509486607e-01, -1.422641718622635e-02, -1.523228224637592e-03, -3.202981425894477e-03, -1.014238760968189e-01, -5.032260791685955e-03, -5.032260791685951e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mspbel_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.807882782061991e+01, -2.807929720214783e+01, -2.807895058086857e+01, -2.807938628978222e+01, -2.807939022559380e+01, -2.808000303297188e+01, -2.807800760976155e+01, -2.807824010298483e+01, -2.807890059767842e+01, -2.807897900527911e+01, -2.807890059767842e+01, -2.807897900527911e+01, -4.206921816228697e+00, -4.206236140282494e+00, -4.207234376504572e+00, -4.206398171237810e+00, -4.212900880510413e+00, -4.213858699144461e+00, -4.212968388039487e+00, -4.212647723700971e+00, -4.205592344350779e+00, -4.217923697499728e+00, -4.205592344350779e+00, -4.217923697499728e+00, -7.929049679789592e-01, -8.008317301146639e-01, -7.902072528876364e-01, -7.996581560986692e-01, -7.537963077675340e-01, -7.440226475469205e-01, -7.605423970966216e-01, -7.650675007753516e-01, -8.115191628189620e-01, -7.073068001153729e-01, -8.115191628189620e-01, -7.073068001153729e-01, -1.470578680117849e-01, -1.852777700035974e-01, -1.551875655692533e-01, -1.828204148596478e-01, -8.301742048265962e-01, -8.843199918077509e-01, -1.475576826786064e-01, -1.457591855191280e-01, -1.616742009107111e-01, -1.292178544478030e-01, -1.616742009107109e-01, -1.292178544478030e-01, -1.299753955345254e-02, -1.380616649740908e-02, -1.361454169715592e-02, -1.457729328846355e-02, -7.021047861139237e-02, -7.346700794960156e-02, -7.828932029715442e-03, -7.698858289904180e-03, -1.160235939794906e-02, -6.610302565233244e-03, -1.160235939794905e-02, -6.610302565233260e-03, -6.916621795529024e+00, -6.915036980563938e+00, -6.918820779952488e+00, -6.917160445194870e+00, -6.916790067651887e+00, -6.915143188208919e+00, -6.918674950702398e+00, -6.917070412064171e+00, -6.917713528855240e+00, -6.916097869828382e+00, -6.917713528855240e+00, -6.916097869828382e+00, -2.244010011428128e+00, -2.250197317444183e+00, -2.300116604950617e+00, -2.304802611903501e+00, -2.237423967300937e+00, -2.244772124163952e+00, -2.292039751377688e+00, -2.299409416408767e+00, -2.283999670175984e+00, -2.280041594672931e+00, -2.283999670175984e+00, -2.280041594672931e+00, -7.504420194800883e-01, -7.485004682445517e-01, -8.669955775615243e-01, -8.667569578922467e-01, -6.522853699951108e-01, -6.989306676722900e-01, -7.130268672149033e-01, -7.649732131959457e-01, -7.881214663489081e-01, -7.474223263792895e-01, -7.881214663489083e-01, -7.474223263792897e-01, -1.328830421972976e-01, -1.329682547552108e-01, -1.497361668351201e-01, -1.543752982726553e-01, -1.263696595535063e-01, -1.290210033814752e-01, -2.588322380856176e+00, -2.587474263784552e+00, -1.319442956155437e-01, -1.295933177851499e-01, -1.319442956155436e-01, -1.295933177851499e-01, -5.875267321427700e-03, -6.105805427268511e-03, -7.532668644200756e-03, -7.646757279504177e-03, -5.630685683458580e-03, -5.961517541196168e-03, -1.008611646887181e-01, -1.015799191000617e-01, -5.753088810362478e-03, -7.893530932540796e-03, -5.753088810362479e-03, -7.893530932540790e-03, -7.926554258434279e-01, -7.957911208153068e-01, -7.833431422292494e-01, -7.865654581670939e-01, -7.865849757076175e-01, -7.897981591765669e-01, -7.893102476950749e-01, -7.924581490161936e-01, -7.879451091883313e-01, -7.911247865095340e-01, -7.879451091883313e-01, -7.911247865095340e-01, -7.739663180220347e-01, -7.764256299667721e-01, -6.226396581371653e-01, -6.259811385130143e-01, -6.633540712904979e-01, -6.669935306348105e-01, -7.069984698055968e-01, -7.096925293581915e-01, -6.843004953161644e-01, -6.873273632040615e-01, -6.843004953161644e-01, -6.873273632040615e-01, -9.119359284714947e-01, -9.132271994770179e-01, -2.048163877843850e-01, -2.053037382199690e-01, -2.459213935500109e-01, -2.509388902885170e-01, -4.024557874425831e-01, -4.054985427781290e-01, -3.166565700014826e-01, -3.171500952723366e-01, -3.166565700014827e-01, -3.171500952723367e-01, -5.344421776678656e-01, -5.421858885952932e-01, -6.931269987842718e-02, -6.971961814407768e-02, -8.814766031019089e-02, -9.019739118620604e-02, -4.029275303061774e-01, -4.118685818348486e-01, -1.120712771456625e-01, -1.111691914310423e-01, -1.120712771456624e-01, -1.111691914310423e-01, -1.854531536260070e-02, -1.920852348905075e-02, -2.028461086233848e-03, -2.032995635349444e-03, -4.126069911340997e-03, -4.388054546615140e-03, -1.081936596143852e-01, -1.057729209044145e-01, -5.447725759220851e-03, -7.237075208007187e-03, -5.447725759220852e-03, -7.237075208007174e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.728443689537277e-09, 0.000000000000000e+00, -8.720696444722655e-09, -8.728232388936188e-09, 0.000000000000000e+00, -8.720544071345100e-09, -8.726553425902781e-09, 0.000000000000000e+00, -8.718421162799019e-09, -8.728888711518516e-09, 0.000000000000000e+00, -8.721407471968581e-09, -8.728338450944367e-09, 0.000000000000000e+00, -8.719481606954900e-09, -8.728338450944367e-09, 0.000000000000000e+00, -8.719481606954900e-09, -1.665820933362183e-05, 0.000000000000000e+00, -1.659380841505692e-05, -1.667838359412534e-05, 0.000000000000000e+00, -1.660218368104494e-05, -1.701973664719164e-05, 0.000000000000000e+00, -1.706686305577282e-05, -1.711978396427076e-05, 0.000000000000000e+00, -1.706873129606325e-05, -1.662911108908822e-05, 0.000000000000000e+00, -1.738703403033253e-05, -1.662911108908822e-05, 0.000000000000000e+00, -1.738703403033253e-05, -1.274014561870044e-02, 0.000000000000000e+00, -1.273273159843195e-02, -1.276239230147224e-02, 0.000000000000000e+00, -1.273786663266037e-02, -1.294060728197781e-02, 0.000000000000000e+00, -1.314034552650608e-02, -1.314587300487242e-02, 0.000000000000000e+00, -1.317906654795145e-02, -1.246401647396990e-02, 0.000000000000000e+00, -1.374744149524125e-02, -1.246401647396990e-02, 0.000000000000000e+00, -1.374744149524125e-02, -3.583203689109071e+00, 0.000000000000000e+00, -2.153431246797316e+00, -3.266664798842982e+00, 0.000000000000000e+00, -2.362601622693123e+00, -5.607554131618436e-03, 0.000000000000000e+00, -5.254935480510582e-03, -2.326428807322068e+00, 0.000000000000000e+00, -2.925699049962458e+00, -2.716386146387366e+00, 0.000000000000000e+00, -3.333544048241338e+00, -2.716386146387370e+00, 0.000000000000000e+00, -3.333544048241345e+00, -8.702626278941247e+00, 0.000000000000000e+00, -8.620560662662177e+00, -9.167563125285811e+00, 0.000000000000000e+00, -9.119490687962623e+00, -4.906040080121706e+00, 0.000000000000000e+00, -4.915096226265703e+00, -7.978294788518348e+00, 0.000000000000000e+00, -7.761891881040371e+00, -8.690228372768457e+00, 0.000000000000000e+00, -2.206169785582582e+01, -8.690228372772356e+00, 0.000000000000000e+00, -2.206169785576413e+01, -1.972833593161210e-06, 0.000000000000000e+00, -1.972332616952600e-06, -1.971707936267928e-06, 0.000000000000000e+00, -1.971248224468313e-06, -1.971817977997174e-06, 0.000000000000000e+00, -1.971607950993178e-06, -1.970889589886533e-06, 0.000000000000000e+00, -1.970659021337613e-06, -1.972978093729201e-06, 0.000000000000000e+00, -1.971908180756845e-06, -1.972978093729201e-06, 0.000000000000000e+00, -1.971908180756845e-06, -2.533875312709824e-04, 0.000000000000000e+00, -2.484237509376935e-04, -2.226387364454948e-04, 0.000000000000000e+00, -2.187168053990702e-04, -2.356227083945257e-04, 0.000000000000000e+00, -2.364617095252992e-04, -2.064275241674421e-04, 0.000000000000000e+00, -2.068687547971175e-04, -2.468756636650491e-04, 0.000000000000000e+00, -2.341169407875290e-04, -2.468756636650491e-04, 0.000000000000000e+00, -2.341169407875290e-04, -4.484507351623889e-02, 0.000000000000000e+00, -4.450739602632466e-02, -3.379064571931275e-02, 0.000000000000000e+00, -3.306468201226447e-02, -3.865363950395681e-02, 0.000000000000000e+00, -5.470593244481130e-02, -3.559811212009831e-02, 0.000000000000000e+00, -4.546297740145910e-02, -3.419050171780749e-02, 0.000000000000000e+00, -4.807295775667280e-02, -3.419050171780748e-02, 0.000000000000000e+00, -4.807295775667279e-02, -2.983133132461645e+00, 0.000000000000000e+00, -2.977080820428197e+00, -2.294719574998943e+00, 0.000000000000000e+00, -2.150591356560604e+00, -3.314948311484310e+00, 0.000000000000000e+00, -3.259466222098074e+00, -3.660355470544410e-04, 0.000000000000000e+00, -3.669912131676864e-04, -3.829803400167222e+00, 0.000000000000000e+00, -4.525565487620645e+00, -3.829803400167227e+00, 0.000000000000000e+00, -4.525565487620632e+00, -1.120186938105397e+01, 0.000000000000000e+00, -9.695595253674703e+00, -9.644066752197874e+00, 0.000000000000000e+00, -8.904186415660243e+00, -5.492446285364867e+01, 0.000000000000000e+00, -6.109246248835296e+01, -5.233237475461040e+00, 0.000000000000000e+00, -5.105037197817965e+00, -2.731771087879738e+01, 0.000000000000000e+00, -2.682161036937305e+01, -2.731771087882345e+01, 0.000000000000000e+00, -2.682161036937908e+01, -1.721074865214983e-02, 0.000000000000000e+00, -1.681906137907462e-02, -1.725507924588780e-02, 0.000000000000000e+00, -1.686923049675417e-02, -1.724152807659343e-02, 0.000000000000000e+00, -1.685140426955681e-02, -1.722733977827389e-02, 0.000000000000000e+00, -1.684080768242449e-02, -1.723761637794396e-02, 0.000000000000000e+00, -1.684816184717596e-02, -1.723761637794396e-02, 0.000000000000000e+00, -1.684816184717597e-02, -2.476767977323632e-02, 0.000000000000000e+00, -2.405867921894262e-02, -3.373756750863254e-02, 0.000000000000000e+00, -3.302495003829471e-02, -3.168485687104502e-02, 0.000000000000000e+00, -3.094660154522272e-02, -2.907307423731646e-02, 0.000000000000000e+00, -2.858969928299517e-02, -3.134955561619702e-02, 0.000000000000000e+00, -3.040327304788752e-02, -3.134955561619703e-02, 0.000000000000000e+00, -3.040327304788755e-02, -2.999562519918549e-02, 0.000000000000000e+00, -2.952163884576537e-02, -1.109952636648854e+00, 0.000000000000000e+00, -1.105610188693981e+00, -7.979918450412253e-01, 0.000000000000000e+00, -7.754568256675057e-01, -2.905820881820460e-01, 0.000000000000000e+00, -2.797140790372489e-01, -5.323052819789653e-01, 0.000000000000000e+00, -5.344186847494126e-01, -5.323052819789658e-01, 0.000000000000000e+00, -5.344186847494126e-01, -1.163338225567079e-01, 0.000000000000000e+00, -1.082562895292562e-01, -4.537728660153482e+00, 0.000000000000000e+00, -4.523387095152153e+00, -4.312907850823898e+00, 0.000000000000000e+00, -4.386637111466362e+00, -3.718567659264424e-01, 0.000000000000000e+00, -4.028561654171950e-01, -5.094436115526067e+00, 0.000000000000000e+00, -6.017178847957513e+00, -5.094436115526078e+00, 0.000000000000000e+00, -6.017178847957518e+00, -6.888258678636229e+00, 0.000000000000000e+00, -7.030399243273935e+00, -3.447502894912072e+01, 0.000000000000000e+00, -6.106585271497575e+01, -2.127412675226961e+01, 0.000000000000000e+00, -2.264688611258913e+01, -5.859988058676858e+00, 0.000000000000000e+00, -8.159233230994747e+00, -5.631302870915057e+01, 0.000000000000000e+00, -2.784793435140698e+01, -5.631302870912177e+01, 0.000000000000000e+00, -2.784793435140202e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mspbel_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mspbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.573300415741785e-04, 1.567306922053678e-04, 1.573206670380240e-04, 1.567239652964770e-04, 1.572168483288998e-04, 1.565960015581339e-04, 1.573187132354053e-04, 1.567267020114739e-04, 1.573258810406997e-04, 1.566205932805512e-04, 1.573258810406997e-04, 1.566205932805512e-04, 1.688111730648614e-03, 1.665445693549995e-03, 1.694938336886693e-03, 1.668198732533623e-03, 1.809679133319921e-03, 1.825540100204627e-03, 1.848604410091887e-03, 1.830964525085473e-03, 1.674996304283212e-03, 1.938563809003153e-03, 1.674996304283212e-03, 1.938563809003153e-03, 1.026753378555437e-02, 1.053931164459792e-02, 1.020831122126957e-02, 1.049111351395365e-02, 9.132038368201562e-03, 9.276994645170508e-03, 1.050966350171549e-02, 1.075024832259593e-02, 1.039905104667628e-02, 1.046408195844635e-02, 1.039905104667628e-02, 1.046408195844635e-02, 9.867135556565729e-02, 5.625353356692859e-02, 8.937735181797736e-02, 6.982380473310827e-02, 1.087854446110532e-03, 1.891680885191709e-03, 6.402683574136757e-03, 2.077485852043100e-02, 7.970850375475304e-02, 1.469858658085674e-04, 7.970850375475320e-02, 1.469858658085701e-04, 6.088531123478588e-10, 1.345309190957873e-08, 2.264490902536149e-10, 2.396192435316137e-09, 1.067644301200276e-06, 2.992407698753870e-06, 3.951278925565041e-15, 5.353318946966612e-15, 7.401687064331140e-12, 1.386958292295511e-15, 7.401687066005869e-12, 1.386958292433616e-15, 4.599140647821348e-05, 4.336303855533785e-05, 4.542950583770443e-05, 4.282287660180012e-05, 4.490556470183028e-05, 4.258430778324964e-05, 4.446474595000000e-05, 4.213360684332331e-05, 4.650128706495995e-05, 4.322448897344922e-05, 4.650128706495995e-05, 4.322448897344922e-05, 1.011065614184175e-02, 9.824762206177305e-03, 8.640071848071480e-03, 8.397783437233030e-03, 8.880664293788452e-03, 8.985795592313828e-03, 7.458822041122912e-03, 7.538511752367179e-03, 1.009225093560894e-02, 9.182534669112574e-03, 1.009225093560894e-02, 9.182534669112574e-03, 5.195704735647870e-02, 5.092505115393982e-02, 4.965872307197330e-02, 4.837867977420323e-02, 2.143689536090027e-02, 5.002118352525444e-02, 2.226993895675734e-02, 4.360894902176387e-02, 4.287618038084808e-02, 5.391120598053534e-02, 4.287618038084805e-02, 5.391120598053533e-02, 1.328992154549989e-03, 1.141630724916919e-03, 6.429880440117310e-02, 5.943535324622724e-02, 3.971213984050299e-04, 1.028879002715615e-03, 1.318441540793384e-02, 1.320753151113608e-02, 1.020055568298236e-02, 1.779509449771099e-02, 1.020055568298239e-02, 1.779509449771086e-02, 8.504912421101344e-18, 1.087037821411386e-17, 4.810296184899677e-16, 3.317626568122942e-16, 2.296936072831607e-14, 4.269723844961205e-14, 2.659916631579349e-05, 1.217686830514455e-04, 2.033104387883460e-17, 1.059684750292730e-11, 2.033104251474805e-17, 1.059684750118529e-11, 7.748623503074583e-03, 7.505763878427955e-03, 7.523213677493407e-03, 7.295039076417278e-03, 7.605065751786747e-03, 7.368846850522408e-03, 7.668494755154842e-03, 7.435318764112397e-03, 7.641705939517984e-03, 7.405254316557227e-03, 7.641705939517984e-03, 7.405254316557232e-03, 1.610022535345812e-02, 1.547594690588217e-02, 1.116237901975112e-02, 1.094428327281192e-02, 1.342677694650517e-02, 1.313068743985231e-02, 1.494842035218726e-02, 1.478907590901508e-02, 1.527901358987152e-02, 1.465002160541167e-02, 1.527901358987154e-02, 1.465002160541171e-02, 5.125758964917376e-02, 5.075090435153731e-02, 6.315346577377527e-02, 6.428471211947595e-02, 8.235942616999463e-02, 8.247210366241342e-02, 6.219622026082985e-02, 6.017323782220424e-02, 8.413254973199377e-02, 8.438054564377781e-02, 8.413254973199383e-02, 8.438054564377778e-02, 6.109154841280828e-02, 5.722371163424582e-02, 1.208324493252097e-05, 6.925721512078565e-06, 6.585874235826568e-06, 1.063493649792198e-05, 7.137074911353475e-02, 8.905127717330699e-02, 6.951482358704508e-04, 1.920138608755242e-03, 6.951482358704500e-04, 1.920138608755244e-03, 5.868133233655692e-13, 6.797191070963124e-13, 9.363198529153807e-22, 9.737930842000396e-22, 1.316936787523874e-15, 2.376278687406273e-15, 1.060034462536932e-03, 7.675074314171418e-03, 1.679293151420396e-16, 4.922295687078534e-12, 1.679293150589246e-16, 4.922295689623937e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05