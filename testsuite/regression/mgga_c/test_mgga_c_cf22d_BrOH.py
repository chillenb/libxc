
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cf22d_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.143531288048816e-01, -1.143540289217004e-01, -1.143567370807205e-01, -1.143420125191585e-01, -1.143536144790392e-01, -1.143536144790392e-01, -8.037287847589104e-03, -8.059599342168908e-03, -8.714152594038162e-03, -8.213819218210695e-03, -8.045010976944204e-03, -8.045010976944204e-03, -1.069941320691774e-02, -1.018288950408565e-02, 3.471095936865314e-03, 6.192034363187901e-03, -1.050923414246785e-02, -1.050923414246785e-02, 1.357418696093855e-02, 1.251493683100561e-02, -6.541679848003525e-02, -1.794351685034850e-02, 1.321950286628989e-02, 1.321950286628989e-02, -1.135031453372476e-02, -1.176922670155901e-02, -2.931386576622181e-02, -6.213369321839195e-03, -1.167024570236675e-02, -1.167024570236675e-02, -1.125253942403314e-01, -1.126868692602470e-01, -1.125384478733650e-01, -1.126641467702886e-01, -1.126136006800497e-01, -1.126136006800497e-01, 3.655998623709295e-02, 3.257719540670168e-02, 3.926711807403916e-02, 3.613828785554018e-02, 3.030248786153687e-02, 3.030248786153687e-02, -3.777047803067897e-02, -8.932802915129710e-02, -2.919695763659955e-02, -5.269749740707933e-02, -4.836758217219309e-02, -4.836758217219309e-02, -2.891399311170287e-02, 1.027242264760821e-02, -2.754687615401367e-02, -7.993794297210290e-02, -1.757140984253258e-02, -1.757140984253258e-02, -6.022771511571494e-03, -6.750395803827418e-03, -5.264254651286397e-03, -3.308503100860485e-02, -6.234917054720070e-03, -6.234917054720067e-03, -1.788451146352033e-01, -1.390177322690871e-01, -1.495256172429494e-01, -1.599280315752229e-01, -1.544036521929163e-01, -1.544036521929163e-01, -1.742717541800708e-01, -2.673132008421237e-02, -4.913214056269937e-02, -8.117470341677598e-02, -6.295366863926659e-02, -6.295366863926659e-02, -7.403399156677579e-02, 2.206680500613331e-02, 1.759956117032632e-02, -9.828125097279438e-03, 4.869374939608536e-03, 4.869374939608593e-03, -1.481321786224541e-02, -2.852753802074861e-02, -3.189217298703051e-02, -2.046029113557794e-02, -2.840391164928248e-02, -2.840391164928240e-02, -1.291145752373314e-02, -2.142121572447499e-03, -3.822908059107193e-03, -2.774423098230967e-02, -5.419547160517962e-03, -5.419547160517972e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cf22d_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.097264393083018e-01, -2.097188584112332e-01, -2.096804727367035e-01, -2.098035289370256e-01, -2.097224946026519e-01, -2.097224946026519e-01, -2.255367467002799e-01, -2.255492423417516e-01, -2.259526955039417e-01, -2.262251312405138e-01, -2.255378270046474e-01, -2.255378270046474e-01, -1.076344938880786e-01, -1.071452601458377e-01, -8.381117837878620e-02, -8.814302868801863e-02, -1.074632338640313e-01, -1.074632338640313e-01, -3.872452268084874e-02, -4.629683785895483e-02, -1.375909381895882e-01, 1.229877795165374e-02, -4.107772873075994e-02, -4.107772873075994e-02, -1.418085029235723e-02, -1.466790152194632e-02, -2.912680205953788e-02, -7.975238125033155e-03, -1.454742040238081e-02, -1.454742040238089e-02, -2.075859236332756e-01, -2.073232773011265e-01, -2.075407372467572e-01, -2.073374726742122e-01, -2.074738932436836e-01, -2.074738932436836e-01, -2.014840249660805e-01, -2.083452931686003e-01, -1.982368630641526e-01, -2.041779673515380e-01, -2.095420819637506e-01, -2.095420819637506e-01, -1.375838099183860e-01, -2.569916238199676e-01, -1.087494149106628e-01, -1.549636139035882e-01, -1.647523679554707e-01, -1.647523679554707e-01, -2.095047603966963e-03, 3.307247332079027e-02, -1.280908565303314e-04, -2.445766002911628e-01, 1.966086861399687e-02, 1.966086861399687e-02, -7.732516204944626e-03, -8.644286076138227e-03, -6.769222570756230e-03, -2.762874273469839e-02, -7.997983961072718e-03, -7.997983961072808e-03, -2.910331324507846e-01, -2.494142691518832e-01, -2.594446643474927e-01, -2.710990352279003e-01, -2.648218619760406e-01, -2.648218619760406e-01, -2.628533857104648e-01, -1.357794709406503e-01, -1.665425247722919e-01, -2.230846564801593e-01, -1.940609156784525e-01, -1.940609156784525e-01, -2.258556749934167e-01, -8.146905101806422e-03, -6.842360497762111e-02, -1.012828619995181e-01, -1.015224788257876e-01, -1.015224788257877e-01, -1.185086536688384e-01, -2.956528655420053e-02, -2.801882634738881e-02, -8.632239093831673e-02, -7.515008843834645e-03, -7.515008843834461e-03, -1.611348233533775e-02, -2.796336388702450e-03, -4.950491853062491e-03, -7.873181825372071e-03, -6.972060647688823e-03, -6.972060647688755e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.396614438508318e-10, 2.396768073068663e-10, 2.397324911379846e-10, 2.394824337312227e-10, 2.396696381098571e-10, 2.396696381098571e-10, 1.749652747113129e-06, 1.749237306332605e-06, 1.735631954120842e-06, 1.729126652000677e-06, 1.749568790016469e-06, 1.749568790016469e-06, 6.623715907239934e-03, 6.592789250572734e-03, 5.563001354404531e-03, 5.326008262544495e-03, 6.612502405551975e-03, 6.612502405551975e-03, 8.693315017285496e-01, 8.892858131392672e-01, 4.492015254557418e-03, 6.030764653556886e-01, 8.771202040419707e-01, 8.771202040419707e-01, 8.188191635617888e-02, 8.828382100628615e-02, 1.929606369530205e-01, 1.524443455146196e-02, 9.201815823515630e-02, 9.201815823515636e-02, 6.468286953959991e-07, 6.579393939707211e-07, 6.480012190569934e-07, 6.566355370375318e-07, 6.524427063348024e-07, 6.524427063348024e-07, 1.380530289110933e-05, 1.300449064599323e-05, 1.370214735849777e-05, 1.308430850056773e-05, 1.335736875828970e-05, 1.335736875828970e-05, 4.706846844148151e-03, 9.669712917303471e-03, 8.653778444633602e-03, 7.058455423004867e-03, 3.945710146696242e-03, 3.945710146696242e-03, 3.319162897705280e-01, 3.077609412543430e-01, 4.135660427011075e-01, 4.190091909830790e-05, 5.743325477641067e-01, 5.743325477641067e-01, 1.571256307905037e-02, 1.957554714761027e-02, 5.926393700719612e-02, 3.249366541880566e-01, 3.023540108302131e-02, 3.023540107757299e-02, 7.734694189571983e-02, 4.830282148410457e-02, 5.698486019471173e-02, 6.490320769129605e-02, 6.079309880181470e-02, 6.079309880181470e-02, 8.074109785987482e-02, 6.577631383799674e-03, 6.678017670914419e-03, 1.618382028385151e-02, 9.685524988646129e-03, 9.685524988646135e-03, 5.441030715652384e-03, 2.111913486696151e-01, 1.621093678741639e-01, 7.808935454258821e-02, 1.180921779048576e-01, 1.180921779048577e-01, 1.744098213804488e-02, 1.314977122519841e-01, 2.135193209892905e-01, 9.290082311676350e-02, 5.365907997563719e-01, 5.365907997563757e-01, 6.763741301206251e-02, 4.910344643881316e-03, 1.194125834335491e-02, 6.934914249121029e-01, 4.089268155691382e-02, 4.089268156890972e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.456834008351335e-06, -1.464530207913680e-06, -1.500366082095716e-06, -1.375284788475104e-06, -1.460867923720120e-06, -1.460867923720120e-06, 8.488828950796185e-04, 8.492956482812370e-04, 8.620541547631298e-04, 8.624235400402147e-04, 8.489572327552518e-04, 8.489572327552518e-04, -8.355482530078954e-03, -8.350488579948492e-03, -8.390487561626416e-03, -7.822038931029403e-03, -8.353009266571162e-03, -8.353009266571162e-03, -5.546567447780959e-02, -5.247515471704841e-02, -6.763577401301435e-03, -4.293349339941815e-02, -5.465514854943332e-02, -5.465514854943332e-02, -2.662025749100202e-04, -3.010328076406257e-04, -5.104223760187300e-03, -1.170008216348762e-05, -3.057442022064356e-04, -3.057442022061744e-04, -9.294761456505016e-04, -9.647180298441008e-04, -9.335185041520962e-04, -9.608919463511945e-04, -9.468777366152541e-04, -9.468777366152541e-04, 2.432732342481762e-03, 2.651651580867058e-03, 2.375400028404146e-03, 2.555341843557807e-03, 2.648499684100262e-03, 2.648499684100262e-03, 2.734555440415935e-02, 8.839439112254106e-02, 1.089137623592771e-02, 3.963488969612289e-02, 3.978305096056745e-02, 3.978305096056745e-02, -2.531380704472600e-02, -5.557243133211901e-02, -2.834772206263357e-02, 6.224617579906982e-03, -4.669026647892406e-02, -4.669026647892406e-02, -1.609868192217068e-05, -1.951687963470082e-05, -3.361795843405442e-05, -8.803633031975762e-03, -1.843469507706448e-05, -1.843469507702010e-05, 2.165295040939329e-01, 3.391280923673588e-02, 5.774211629425453e-02, 1.010822864879159e-01, 7.582886255363101e-02, 7.582886255363101e-02, 1.498174903277919e-01, 3.637731587591161e-02, 5.687015911723148e-02, 8.252269823428356e-02, 7.098688264718796e-02, 7.098688264718797e-02, 6.149635125746861e-02, -3.997994405275691e-02, -1.826872095574366e-02, 2.333237794509635e-03, -4.180576170072312e-04, -4.180576170071988e-04, 1.890018081552705e-02, -3.882174804017719e-03, -7.669456027334910e-03, 7.209105729634516e-03, -2.613643809101789e-02, -2.613643809101750e-02, -1.620586304423402e-04, -5.118419964561812e-07, -6.486265673390717e-06, -2.775714860073156e-02, -1.626236859922154e-05, -1.626236859923331e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05