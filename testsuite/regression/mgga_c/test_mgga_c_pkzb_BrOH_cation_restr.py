
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_pkzb_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.453914500773632e-02, -5.453990395154163e-02, -5.454232893603387e-02, -5.453107246704272e-02, -5.453710504149963e-02, -5.453710504149963e-02, -4.222950588086458e-02, -4.223465885810211e-02, -4.235545260886278e-02, -4.216168550776402e-02, -4.224316080303092e-02, -4.224316080303092e-02, -2.785887818364394e-02, -2.763211177962079e-02, -2.221417883092519e-02, -2.255228865788398e-02, -2.266241016877890e-02, -2.266241016877890e-02, -7.616779208770246e-03, -8.244405597084302e-03, -3.007123840233906e-02, -2.325267111210512e-03, -4.117869249968382e-03, -4.117869249968382e-03, -6.979630969248376e-09, -9.309412228034961e-09, -1.039505311841879e-05, -4.722297839375342e-10, -1.775241342054395e-09, -1.775241342054394e-09, -6.051404131909300e-02, -6.068072634304328e-02, -6.052140753339852e-02, -6.066852991158413e-02, -6.059862455599651e-02, -6.059862455599651e-02, -2.405834266814883e-02, -2.467904852578006e-02, -2.309186386382229e-02, -2.363973336800753e-02, -2.488426636321945e-02, -2.488426636321945e-02, -3.939874270420781e-02, -5.535158356290277e-02, -3.665370526581720e-02, -5.110403290992255e-02, -4.109561691956880e-02, -4.109561691956880e-02, -4.752614191297155e-04, -3.922245955364917e-03, -3.693481158706963e-04, -7.160605358101578e-02, -1.279369186502496e-03, -1.279369186502497e-03, -1.681973851739169e-10, -4.771623597285612e-10, -8.825899222995123e-10, -9.544499431587549e-05, -9.571448609502734e-10, -9.571448609502738e-10, -5.722343173936966e-02, -5.118132593720023e-02, -5.277406457810439e-02, -5.443575871733781e-02, -5.355608317584253e-02, -5.355608317584253e-02, -6.099651744753517e-02, -3.024802564471689e-02, -3.643126514710830e-02, -4.326517470057662e-02, -3.972583436596462e-02, -3.972583436596461e-02, -5.540653397483297e-02, -7.035923249442548e-03, -1.177480559856954e-02, -2.481162516289865e-02, -1.786384243152332e-02, -1.786384243152334e-02, -2.840814213133354e-02, -8.020597951679100e-06, -2.974931582714200e-05, -2.933032314491652e-02, -3.177554066426791e-04, -3.177554066426783e-04, -2.476473365973729e-08, -3.964660771496456e-12, -7.886497156261795e-11, -2.477540354308021e-04, -8.552388227692000e-10, -8.552388248104812e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_pkzb_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.147538056748202e-01, -1.147545780851983e-01, -1.147558602171931e-01, -1.147443627422998e-01, -1.147506936147564e-01, -1.147506936147564e-01, -1.068548994408642e-01, -1.068603121693346e-01, -1.069817204411619e-01, -1.067300270285798e-01, -1.068416497735564e-01, -1.068416497735564e-01, -7.751634488657866e-02, -7.723585345918532e-02, -6.937313332342868e-02, -6.977764162088458e-02, -7.000895049219538e-02, -7.000895049219538e-02, -2.985069959085252e-02, -3.161242721894611e-02, -8.274577470664202e-02, -1.149050903977018e-02, -1.866803420791701e-02, -1.866803420791700e-02, -4.246455748884806e-08, -5.663879903947553e-08, -6.157206566144308e-05, -2.926537624232647e-09, -1.092069645040255e-08, -1.092069644916118e-08, -1.128760032434117e-01, -1.130292092527502e-01, -1.128780035824108e-01, -1.130134105654722e-01, -1.129561734867349e-01, -1.129561734867349e-01, -7.353714486043200e-02, -7.434880126488028e-02, -7.172798942846115e-02, -7.247550950979652e-02, -7.486107424057001e-02, -7.486107424057001e-02, -7.895901112069853e-02, -8.264392003702850e-02, -7.861063704688658e-02, -8.090892924537746e-02, -7.966100965934085e-02, -7.966100965934085e-02, -2.619797841119585e-03, -1.789791753202764e-02, -2.058697296739469e-03, -1.174317316649173e-01, -6.612176462382867e-03, -6.612176462382868e-03, -1.070611572413820e-09, -2.994245359551494e-09, -5.495496502015206e-09, -5.521708276065502e-04, -5.996893791759246e-09, -5.996893794597333e-09, -7.664772571943930e-02, -7.606702570745007e-02, -7.667868747412473e-02, -7.698426252685983e-02, -7.686908262420201e-02, -7.686908262420200e-02, -7.358709748018583e-02, -6.694859100830071e-02, -6.961652646967202e-02, -7.257025653970484e-02, -7.106923346035764e-02, -7.106923346035764e-02, -8.469371071479898e-02, -2.871159288402465e-02, -4.167762286300826e-02, -6.295486791914869e-02, -5.326955508516983e-02, -5.326955508516985e-02, -6.775119108610600e-02, -4.729601956687944e-05, -1.746603793053754e-04, -6.460206660286506e-02, -1.773447770620936e-03, -1.773447770620937e-03, -1.537674257755509e-07, -2.590316298918603e-11, -4.895526225966520e-10, -1.386040984139709e-03, -5.336036991206147e-09, -5.336036997821516e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.219729036469113e-10, 1.219755870549334e-10, 1.219747007427263e-10, 1.219347624294610e-10, 1.219575115032947e-10, 1.219575115032947e-10, 9.834280428445991e-07, 9.836202411991991e-07, 9.877967834678366e-07, 9.778694876737408e-07, 9.827006142188869e-07, 9.827006142188869e-07, 1.686074582543749e-03, 1.672473220940290e-03, 1.334899218230998e-03, 1.292194343219639e-03, 1.321062942265757e-03, 1.321062942265757e-03, 1.256656123056641e-01, 1.315002554726905e-01, 9.553924358775371e-04, 1.057245605881626e-01, 1.302864972446739e-01, 1.302864972446738e-01, 4.258824026541421e-03, 5.017609712912871e-03, 2.565088927893152e-02, 1.504872373706850e-03, 3.145387580576448e-03, 3.145387583020710e-03, 1.825723080772860e-07, 1.843890963375928e-07, 1.826188870727084e-07, 1.842233781738846e-07, 1.835081881512471e-07, 1.835081881512471e-07, 4.414146382372760e-06, 4.317125200595599e-06, 4.185369607553712e-06, 4.100496512800239e-06, 4.476038819446340e-06, 4.476038819446340e-06, 4.784875332022084e-03, 8.042950369653830e-03, 6.705353093569911e-03, 1.158335415172469e-02, 4.870881472069329e-03, 4.870881472069329e-03, 6.132492135153462e-02, 5.051448614035288e-02, 6.272371044188250e-02, 6.086295216009007e-05, 1.145662967978332e-01, 1.145662967978332e-01, 1.469887142979253e-03, 1.850094725701631e-03, 1.851132327235470e-02, 6.176447344457867e-02, 8.056253486214426e-03, 8.056253492141376e-03, 3.423469962719382e-02, 1.723152948685090e-02, 2.166985770887672e-02, 2.659436494886139e-02, 2.396921586497751e-02, 2.396921586497749e-02, 2.613856323033080e-02, 4.256628277565576e-03, 5.160622491067439e-03, 8.355975899361160e-03, 6.336753906282604e-03, 6.336753906282605e-03, 6.190122665643802e-03, 3.509905330057687e-02, 3.026846057446936e-02, 2.678294193228447e-02, 2.843753468809882e-02, 2.843753468809886e-02, 7.896202608455248e-03, 2.096802422941118e-02, 3.211415505026442e-02, 4.226044714287444e-02, 1.069941886472029e-01, 1.069941886472018e-01, 5.290687511588656e-03, 2.005775744416231e-03, 2.523784717637883e-03, 9.780340862864138e-02, 1.003184901443337e-02, 1.003184900264868e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.990863561402912e-06, 6.990867350605859e-06, 6.991586857168368e-06, 6.991558454167782e-06, 6.991473036682121e-06, 6.991473036682121e-06, 2.130683546702497e-05, 2.131197898286416e-05, 2.148440560808613e-05, 2.177086566462564e-05, 2.159286337535568e-05, 2.159286337535568e-05, 1.530879828963199e-04, 1.548936165808278e-04, 1.955897727623204e-04, 2.307412948298334e-04, 2.173608158204437e-04, 2.173608158204437e-04, 4.607578631752967e-03, 4.904813427373382e-03, 1.723314997166696e-05, 8.217201224433406e-04, 1.818699822367721e-03, 1.818699822367721e-03, 4.270486442773064e-09, 5.774687694677081e-09, 4.328804648326808e-06, 1.999285317873282e-10, 9.723083885943048e-10, 9.723083885943025e-10, 9.005119414832441e-05, 8.857972340807080e-05, 9.002653986988414e-05, 8.872632575314840e-05, 8.928954875400519e-05, 8.928954875400519e-05, 2.197594623649198e-04, 2.363187335852710e-04, 2.187513705214856e-04, 2.341021593635413e-04, 2.290679670697731e-04, 2.290679670697731e-04, 8.432882999033774e-04, -1.428688128284510e-03, 5.823899048520468e-04, -3.677193056211251e-04, 5.702004163250840e-04, 5.702004163250840e-04, 1.686946075865578e-04, 1.557111317493833e-03, 1.317056673213924e-04, -3.031911417877020e-05, 6.309151378758944e-04, 6.309151378758946e-04, 3.329617992618578e-11, 1.491294839008642e-10, 9.492206214674918e-10, 3.889412875075755e-05, 5.507270834035367e-10, 5.507270834035379e-10, -4.571171962619592e-02, -2.742093581673288e-02, -3.507255027951044e-02, -4.160361011484477e-02, -3.839314880384279e-02, -3.839314880384279e-02, -8.395329756166688e-03, 5.222636999308315e-03, 4.159026438877526e-03, -1.708731790210694e-03, 2.130413692876687e-03, 2.130413692876685e-03, -7.875083722843578e-04, 2.631723700072848e-03, 3.586298515493474e-03, 2.930537293655588e-03, 4.309243168886409e-03, 4.309243168886413e-03, 3.169725198004427e-03, 3.670388240823428e-06, 1.140185215153711e-05, 2.914427047231837e-03, 1.576097040136060e-04, 1.576097040136123e-04, 7.025688740755250e-09, 3.412052165806381e-13, 6.111160338709324e-11, 1.397296107616425e-04, 6.463475735966276e-10, 6.463475730116071e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05