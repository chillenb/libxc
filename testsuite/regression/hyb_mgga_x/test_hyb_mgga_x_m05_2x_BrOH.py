
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m05_2x_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.828132867506329e+00, -7.828126847695156e+00, -7.828129950602582e+00, -7.828226389869469e+00, -7.828129462185771e+00, -7.828129462185771e+00, -1.592487838271432e+00, -1.592492643013534e+00, -1.592717813375569e+00, -1.593379688489299e+00, -1.592492772952552e+00, -1.592492772952552e+00, -3.582846760957946e-01, -3.591161098794441e-01, -3.813608339688909e-01, -3.747974812897759e-01, -3.585715149437660e-01, -3.585715149437660e-01, -1.185660588624153e-01, -1.170470262033045e-01, -4.734960450584865e-01, -1.342737190493760e-01, -1.181354704084674e-01, -1.181354704084674e-01, -2.780529245210889e-02, -2.908255941038773e-02, -1.014586371101686e-01, -1.351569757738913e-02, -2.877903504963621e-02, -2.877903504963620e-02, -1.798537694811405e+00, -1.795803312850588e+00, -1.798240442243733e+00, -1.796115750167743e+00, -1.797163480756581e+00, -1.797163480756581e+00, -9.560833338637883e-01, -9.621994252741157e-01, -9.563006741902330e-01, -9.606926549360062e-01, -9.604734217202352e-01, -9.604734217202352e-01, -2.470365040195116e-01, -2.227476377548579e-01, -2.495838643860130e-01, -2.235917272549646e-01, -2.385523697803247e-01, -2.385523697803247e-01, -1.419799878151271e-01, -1.460725931714171e-01, -1.362091874526880e-01, -7.258474561827528e-01, -1.251127895491252e-01, -1.251127895491252e-01, -1.303526474009654e-02, -1.488935203038124e-02, -1.115693936647906e-02, -1.257548684603405e-01, -1.356898987033820e-02, -1.356898987033827e-02, -2.680871824202034e-01, -2.104809632261675e-01, -2.300328282182485e-01, -2.485712995441588e-01, -2.391511493953814e-01, -2.391511493953814e-01, -2.400536013133218e-01, -2.273650460490336e-01, -2.045378417324295e-01, -1.959054265007029e-01, -1.986359718221073e-01, -1.986359718221073e-01, -2.388146875449592e-01, -1.597456187036159e-01, -1.581761391546515e-01, -1.660535380145021e-01, -1.548688673002805e-01, -1.548688673002805e-01, -2.154648992574481e-01, -9.684991714987518e-02, -1.182894532770822e-01, -1.545040879857294e-01, -1.245549846483915e-01, -1.245549846483914e-01, -3.263123306024426e-02, -4.087743635169566e-03, -7.758158732233005e-03, -1.188694700529354e-01, -1.153642451398159e-02, -1.153642451398159e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m05_2x_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.473757314147564e+00, -8.473781483916556e+00, -8.473889052941919e+00, -8.473490723944980e+00, -8.473770073088501e+00, -8.473770073088501e+00, -1.982893917535983e+00, -1.982624462073211e+00, -1.974218078936767e+00, -1.977247643807992e+00, -1.982831583312999e+00, -1.982831583312999e+00, -1.539748835270135e-01, -1.539747199743194e-01, -1.739345495614236e-01, -1.556367299972621e-01, -1.539582560439784e-01, -1.539582560439784e-01, -4.611768704396843e-02, -4.189693003961335e-02, -3.620745741603427e-01, -3.695521609082487e-02, -4.483353554485323e-02, -4.483353554485323e-02, -3.629107741267285e-02, -3.788095631545749e-02, -1.045141674246021e-01, -1.796362749323012e-02, -3.748500081465718e-02, -3.748500081465985e-02, -1.678331304605985e+00, -1.673760439641213e+00, -1.677789851922844e+00, -1.674238332024382e+00, -1.675983089116297e+00, -1.675983089116297e+00, -1.044263028841854e+00, -1.088862801425059e+00, -1.020858450312713e+00, -1.061187905710960e+00, -1.098470410797108e+00, -1.098470410797108e+00, -1.577867675137596e-01, -2.603173602195754e-01, -2.463505443710692e-01, -1.584694795977601e-01, -1.593979270802978e-01, -1.593979270802978e-01, -5.349246423137750e-02, -9.937422950290964e-02, -4.796458093072443e-02, -5.625248687023365e-01, -3.439664350720816e-02, -3.439664350720816e-02, -1.732010042963117e-02, -1.976846372790444e-02, -1.480880529075263e-02, -1.043628394163439e-01, -1.801948962857954e-02, -1.801948962857903e-02, -4.033944163977282e-01, -4.223860223359925e-01, -4.772897130657813e-01, -4.824954289519565e-01, -4.854710876814311e-01, -4.854710876814311e-01, -4.684565357439409e-01, -1.445106052910544e-01, -1.547829710233426e-01, -2.101883835271789e-01, -2.036282484252401e-01, -2.036282484252401e-01, -2.439731554097652e-01, -7.685830933824765e-02, -5.185561079037831e-02, -1.933592926898380e-01, -1.187899620115856e-01, -1.187899620115857e-01, -2.401923610818996e-01, -1.049236060067081e-01, -1.073058797980174e-01, -1.823553254860271e-01, -5.694745896899479e-02, -5.694745896898128e-02, -4.264690803962216e-02, -5.448375831805575e-03, -1.032764129681287e-02, -5.430719192165207e-02, -1.532879973069496e-02, -1.532879973069454e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.635586946714139e-09, -2.635565930557586e-09, -2.635439910654766e-09, -2.635784184503018e-09, -2.635576139849180e-09, -2.635576139849180e-09, -4.363077111779596e-06, -4.363226742353145e-06, -4.366693871371026e-06, -4.358970638962488e-06, -4.363067888884746e-06, -4.363067888884746e-06, -3.113996964523381e-03, -3.122069988765969e-03, -3.280424203247844e-03, -3.126203623955755e-03, -3.116877648566633e-03, -3.116877648566633e-03, -3.518973023662796e-01, -3.434546622480312e-01, -1.883593763920298e-03, -7.151594601491973e-01, -3.497111671862616e-01, -3.497111671862616e-01, -8.378959031340314e+00, -8.391429126454957e+00, -3.084993016594606e+00, -6.001389978182065e+00, -8.717113115941466e+00, -8.717113115941462e+00, -7.432085233506381e-07, -7.422415458383169e-07, -7.430983700958554e-07, -7.423471652215486e-07, -7.427301184112579e-07, -7.427301184112579e-07, -3.318883522956924e-05, -3.263165986391050e-05, -3.311052979623501e-05, -3.266754458352850e-05, -3.291866639866049e-05, -3.291866639866049e-05, -5.436159364203782e-03, -3.847750300116520e-03, -7.030705926308088e-03, -5.945622169721967e-03, -4.474604058368157e-03, -4.474604058368157e-03, -1.118810996953456e+00, -2.368663097750017e-01, -1.249660952644484e+00, -4.813763720261645e-05, -9.424577617373856e-01, -9.424577617373856e-01, -6.364904124402719e+00, -6.375575885004093e+00, -1.824320578523603e+01, -2.742944706442390e+00, -9.418817139167187e+00, -9.418817139167235e+00, -6.258069372333025e-03, -4.864074330036721e-03, -5.334979209715676e-03, -5.781159859063641e-03, -5.554266287387903e-03, -5.554266287387903e-03, -6.636816086718851e-03, -8.778577441943394e-03, -7.197914802557339e-03, -6.313339660863714e-03, -6.705925469612280e-03, -6.705925469612279e-03, -3.236987852848434e-03, -1.236680234586559e-01, -7.407474172442073e-02, -3.604704112307040e-02, -5.075811749588013e-02, -5.075811749588014e-02, -1.356914191468101e-02, -2.675150684350424e+00, -2.446155550972418e+00, -5.039291291681653e-02, -2.046043010977042e+00, -2.046043010977043e+00, -6.306339270841538e+00, -1.099060849909540e+01, -9.455913562448185e+00, -2.514065816220787e+00, -1.375983035520749e+01, -1.375983035520749e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.410759756560953e-04, -5.410822949791860e-04, -5.411099906052967e-04, -5.410084411136596e-04, -5.410793008565136e-04, -5.410793008565136e-04, 9.951689823143107e-04, 9.930715332947255e-04, 9.271101682670004e-04, 9.536035086188423e-04, 9.945801449816587e-04, 9.945801449816587e-04, -2.170388755835304e-02, -2.164969722829452e-02, -1.858825092919651e-02, -2.063716928521226e-02, -2.168650901705499e-02, -2.168650901705499e-02, -7.048017586197368e-02, -7.554232758961744e-02, -7.470795905460912e-03, -5.633228389221376e-02, -7.197072979056092e-02, -7.197072979056092e-02, -6.983438346940721e-04, -7.956234057602859e-04, -1.647112714245370e-02, -2.756157847328351e-05, -8.064066136534430e-04, -8.064066136512853e-04, -7.724976228113602e-03, -7.817985474879932e-03, -7.735640898123210e-03, -7.807915437589374e-03, -7.772469277445022e-03, -7.772469277445022e-03, 1.092139395115841e-03, 1.814001331375117e-03, 7.127503629095719e-04, 1.375044827789524e-03, 1.958894930495156e-03, 1.958894930495156e-03, -5.670233697758992e-02, -1.760358388951943e-02, -1.701420913736457e-02, -6.257449686959847e-02, -6.005856762197648e-02, -6.005856762197648e-02, -6.038848980327413e-02, -2.393724679289135e-02, -6.422200601770636e-02, -1.780650099848674e-02, -6.883794051738429e-02, -6.883794051738429e-02, -3.772272895738487e-05, -4.658870424021796e-05, -7.708747552968871e-05, -2.800722682524728e-02, -4.342690323557880e-05, -4.342690323618494e-05, 2.240487254294326e-01, 3.444065295441163e-01, 4.944077296961296e-01, 5.233005270804064e-01, 5.250844092861937e-01, 5.250844092861937e-01, 5.357782596186329e-01, -5.637707210324917e-02, -5.858056447286723e-02, -3.044797920411860e-02, -2.795880058543431e-02, -2.795880058543428e-02, -3.094073778196488e-02, -4.340884539249035e-02, -5.968189574172518e-02, 8.534560569952710e-03, -2.455409335320572e-02, -2.455409335320580e-02, 1.664048383681577e-03, -1.262938727810401e-02, -2.465624203384786e-02, 6.593957628425060e-03, -6.413838608671367e-02, -6.413838608670400e-02, -4.394234335783705e-04, -1.059405749742761e-06, -1.426631992229246e-05, -6.692751370148334e-02, -3.748319412853523e-05, -3.748319412906206e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05