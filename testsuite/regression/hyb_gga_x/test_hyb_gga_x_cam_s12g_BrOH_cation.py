
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_cam_s12g_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.060047227048349e+01, -2.060050281832925e+01, -2.060070118338981e+01, -2.060024545923083e+01, -2.060047533745943e+01, -2.060047533745943e+01, -3.175000933938976e+00, -3.174960155400179e+00, -3.174103604249796e+00, -3.176248781956378e+00, -3.175035366143899e+00, -3.175035366143899e+00, -5.138539297589324e-01, -5.139834985614280e-01, -5.187160408000275e-01, -5.225957913285578e-01, -5.216200003677207e-01, -5.216200003677207e-01, -1.497445990477657e-01, -1.506044091911498e-01, -6.052891796951385e-01, -1.242598907784964e-01, -1.459673863606761e-01, -1.459673863606761e-01, -6.450090928375975e-03, -6.792160052456503e-03, -3.710900090370465e-02, -3.719732785268293e-03, -5.191494437799481e-03, -5.191494437799482e-03, -4.747972768501055e+00, -4.748011467011060e+00, -4.747982023953364e+00, -4.748015914220229e+00, -4.747987013548056e+00, -4.747987013548056e+00, -1.860378634895065e+00, -1.869494066262642e+00, -1.863194566172839e+00, -1.871329669403106e+00, -1.863961543539690e+00, -1.863961543539690e+00, -4.153975437538415e-01, -4.441615237242217e-01, -3.842167375885132e-01, -3.877636961726832e-01, -4.210815394730831e-01, -4.210815394730831e-01, -9.214381383560796e-02, -1.594092674968393e-01, -8.587597360944818e-02, -1.584925852895590e+00, -1.045701451702958e-01, -1.045701451702958e-01, -2.871839444195310e-03, -3.636249471479201e-03, -2.784530227143691e-03, -5.908026786006393e-02, -3.497773114750784e-03, -3.497773114750784e-03, -4.060277619126913e-01, -4.038149734124166e-01, -4.045257928472674e-01, -4.051696825153244e-01, -4.048411279099371e-01, -4.048411279099371e-01, -3.935801613026783e-01, -3.618254803656679e-01, -3.646296718973238e-01, -3.712894518540362e-01, -3.669923504427540e-01, -3.669923504427540e-01, -4.689499567659495e-01, -1.923765932525169e-01, -2.179307911285062e-01, -2.532980867200407e-01, -2.348672581183324e-01, -2.348672581183324e-01, -3.315121443542247e-01, -3.551014463831423e-02, -4.809367994084984e-02, -2.337116742679100e-01, -7.377267382631542e-02, -7.377267382631544e-02, -9.089619378021024e-03, -9.719804524421234e-04, -2.044003180306450e-03, -6.961081710513667e-02, -3.212157605271699e-03, -3.212157605271696e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_cam_s12g_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.507584246613889e+01, -2.507581935400281e+01, -2.507604353660253e+01, -2.507596535532631e+01, -2.507654350858867e+01, -2.507672189494993e+01, -2.507427637791259e+01, -2.507383106772118e+01, -2.507596962453452e+01, -2.507488124069071e+01, -2.507596962453452e+01, -2.507488124069071e+01, -3.691989542783917e+00, -3.692717912878467e+00, -3.692099592491915e+00, -3.692870261173524e+00, -3.695199822279349e+00, -3.696037054053481e+00, -3.690461515423757e+00, -3.691374461235458e+00, -3.691028953980476e+00, -3.694180745366074e+00, -3.691028953980476e+00, -3.694180745366074e+00, -5.402953247266868e-01, -5.450636082420944e-01, -5.386682856840608e-01, -5.443527841150296e-01, -5.227249593768003e-01, -5.197398644216282e-01, -5.255800231432038e-01, -5.265628850241121e-01, -5.544655568672722e-01, -5.227443393836754e-01, -5.544655568672722e-01, -5.227443393836754e-01, -1.441019935228570e-01, -1.458484526237716e-01, -1.436149175836841e-01, -1.457373534406947e-01, -6.386275843155236e-01, -6.795329452677838e-01, -1.327184285769068e-01, -1.339844729692117e-01, -1.495721568260312e-01, -1.104541316510603e-01, -1.495721568260312e-01, -1.104541316510603e-01, -8.310670454565490e-03, -8.829970476916455e-03, -8.707469759324583e-03, -9.326362091827800e-03, -4.721060686250082e-02, -4.969775662893428e-02, -4.999072734740459e-03, -4.915852988156408e-03, -7.415444223662107e-03, -4.221485758467345e-03, -7.415444223662108e-03, -4.221485758467346e-03, -6.127593192348611e+00, -6.125897299922749e+00, -6.132833814723504e+00, -6.130962415150541e+00, -6.127873618991003e+00, -6.126063573890931e+00, -6.132371103951722e+00, -6.130665536830847e+00, -6.130290739477433e+00, -6.128446826848811e+00, -6.130290739477433e+00, -6.128446826848811e+00, -1.856988638644054e+00, -1.856893453074941e+00, -1.869418257583317e+00, -1.868918961328286e+00, -1.852495734415546e+00, -1.853399591396518e+00, -1.862500408391008e+00, -1.863804894322098e+00, -1.869143115842010e+00, -1.863776764879072e+00, -1.869143115842010e+00, -1.863776764879072e+00, -5.083359038531883e-01, -5.069763460194953e-01, -6.089366469638854e-01, -6.094799876326135e-01, -4.413176470754607e-01, -4.669595006700576e-01, -5.159349053819630e-01, -5.354067740792356e-01, -5.424835481356657e-01, -5.079155509510125e-01, -5.424835481356657e-01, -5.079155509510125e-01, -1.098353417570934e-01, -1.103504274192765e-01, -1.644174016258860e-01, -1.648585673366912e-01, -1.016161389125391e-01, -1.055853653887221e-01, -2.170596451854332e+00, -2.169552188346047e+00, -1.177117176567346e-01, -1.177790465793643e-01, -1.177117176567346e-01, -1.177790465793643e-01, -3.750685890291472e-03, -3.897867668530003e-03, -4.809956983441183e-03, -4.882773624757569e-03, -3.596319338413376e-03, -3.808296540323848e-03, -7.444614615068146e-02, -7.494444437007461e-02, -3.673558596205831e-03, -5.043438143372993e-03, -3.673558596205831e-03, -5.043438143372995e-03, -5.576836669189489e-01, -5.601977506418396e-01, -5.516922453084938e-01, -5.543109414301239e-01, -5.541654378183122e-01, -5.567482529909183e-01, -5.559016771133004e-01, -5.584289506713815e-01, -5.550679370338429e-01, -5.576202039803811e-01, -5.550679370338429e-01, -5.576202039803811e-01, -5.405480126393566e-01, -5.426165588713441e-01, -3.765198638529841e-01, -3.788821164783140e-01, -4.207792707487464e-01, -4.242062726150493e-01, -4.804015177507780e-01, -4.827581874019097e-01, -4.503116621979901e-01, -4.527727592742810e-01, -4.503116621979901e-01, -4.527727592742810e-01, -6.420948587117502e-01, -6.437311391387368e-01, -1.887629441173984e-01, -1.892277346436735e-01, -2.055760091025368e-01, -2.061906216231125e-01, -2.504695838323538e-01, -2.521300277801509e-01, -2.206308894133829e-01, -2.203052359839906e-01, -2.206308894133828e-01, -2.203052359839906e-01, -3.406613543745066e-01, -3.440984497076572e-01, -4.642595585872827e-02, -4.672719064204744e-02, -6.123396966897031e-02, -6.305840570017640e-02, -2.465356370428784e-01, -2.521907662818549e-01, -8.835612480671630e-02, -9.077859058287296e-02, -8.835612480671630e-02, -9.077859058287301e-02, -1.187811696457824e-02, -1.230682771936379e-02, -1.294450907547689e-03, -1.297388012588934e-03, -2.633619124853208e-03, -2.800992215875791e-03, -8.498245677067752e-02, -8.613215319424709e-02, -3.479286182831753e-03, -4.623118153689723e-03, -3.479286182831748e-03, -4.623118153689718e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_cam_s12g_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.263486506488561e-08, 0.000000000000000e+00, -1.263489885395143e-08, -1.263431506286319e-08, 0.000000000000000e+00, -1.263449951942413e-08, -1.263304191053074e-08, 0.000000000000000e+00, -1.263253899776559e-08, -1.263924065243194e-08, 0.000000000000000e+00, -1.264044380491530e-08, -1.263451247156565e-08, 0.000000000000000e+00, -1.263764172449723e-08, -1.263451247156565e-08, 0.000000000000000e+00, -1.263764172449723e-08, -2.016134473595601e-05, 0.000000000000000e+00, -2.015009992622056e-05, -2.015956131311633e-05, 0.000000000000000e+00, -2.014767582061767e-05, -2.010958525789598e-05, 0.000000000000000e+00, -2.009591600695529e-05, -2.018314806518764e-05, 0.000000000000000e+00, -2.016865322614471e-05, -2.017927055362490e-05, 0.000000000000000e+00, -2.012337374908106e-05, -2.017927055362490e-05, 0.000000000000000e+00, -2.012337374908106e-05, -1.189171543711423e-02, 0.000000000000000e+00, -1.186551074082479e-02, -1.190161738110792e-02, 0.000000000000000e+00, -1.187966340712677e-02, -1.173390803563510e-02, 0.000000000000000e+00, -1.151356006869064e-02, -1.131108932996489e-02, 0.000000000000000e+00, -1.138470466509344e-02, -1.166935532757392e-02, 0.000000000000000e+00, -9.743613728534870e-03, -1.166935532757392e-02, 0.000000000000000e+00, -9.743613728534870e-03, -8.322739835487699e-01, 0.000000000000000e+00, -8.093634171094353e-01, -8.527962374586175e-01, 0.000000000000000e+00, -8.216719038531014e-01, -6.973537259570962e-03, 0.000000000000000e+00, -6.272990769835151e-03, -8.983698062961658e-01, 0.000000000000000e+00, -8.740029526599479e-01, -7.240745033412215e-01, 0.000000000000000e+00, -1.114295490367449e+00, -7.240745033412211e-01, 0.000000000000000e+00, -1.114295490367450e+00, -1.840849812081939e+00, 0.000000000000000e+00, -1.822626497905991e+00, -1.939856907744297e+00, 0.000000000000000e+00, -1.930253163106415e+00, -1.102406615040364e+00, 0.000000000000000e+00, -1.111807582147635e+00, -1.685018783369078e+00, 0.000000000000000e+00, -1.639250499747807e+00, -1.837388863825594e+00, 0.000000000000000e+00, -4.660169178687861e+00, -1.837388863825598e+00, 0.000000000000000e+00, -4.660169178687874e+00, -2.390133970292710e-06, 0.000000000000000e+00, -2.393324371940448e-06, -2.365066809719992e-06, 0.000000000000000e+00, -2.369075580081646e-06, -2.388834910954489e-06, 0.000000000000000e+00, -2.392558119968082e-06, -2.367324595154906e-06, 0.000000000000000e+00, -2.370528241118678e-06, -2.377219277769029e-06, 0.000000000000000e+00, -2.381130887083573e-06, -2.377219277769029e-06, 0.000000000000000e+00, -2.381130887083573e-06, -1.637694955744930e-04, 0.000000000000000e+00, -1.637945137598221e-04, -1.616027661104804e-04, 0.000000000000000e+00, -1.617151693264876e-04, -1.609966466420676e-04, 0.000000000000000e+00, -1.619083174550541e-04, -1.593130589968480e-04, 0.000000000000000e+00, -1.600986246576006e-04, -1.642136348003906e-04, 0.000000000000000e+00, -1.629755640860327e-04, -1.642136348003906e-04, 0.000000000000000e+00, -1.629755640860327e-04, -1.737118477675856e-02, 0.000000000000000e+00, -1.751937939054580e-02, -2.564351406186235e-03, 0.000000000000000e+00, -2.448830970329021e-03, -2.815475589005998e-02, 0.000000000000000e+00, -2.334097733102975e-02, -6.733457091674542e-03, 0.000000000000000e+00, -6.169269351099134e-03, -1.341001552479658e-02, 0.000000000000000e+00, -1.702057566546949e-02, -1.341001552479659e-02, 0.000000000000000e+00, -1.702057566546948e-02, -8.584841329522483e-01, 0.000000000000000e+00, -8.762571022442947e-01, -4.434962420417380e-01, 0.000000000000000e+00, -4.404413933167370e-01, -9.488992863374678e-01, 0.000000000000000e+00, -9.324641431453814e-01, -3.324183026347006e-05, 0.000000000000000e+00, -3.321812907750450e-05, -9.933016711364344e-01, 0.000000000000000e+00, -1.246381500215657e+00, -9.933016711364344e-01, 0.000000000000000e+00, -1.246381500215657e+00, -2.365174725114506e+00, 0.000000000000000e+00, -2.047143900615592e+00, -2.036861083452969e+00, 0.000000000000000e+00, -1.880579891226988e+00, -1.160357040209901e+01, 0.000000000000000e+00, -1.290935067909975e+01, -1.330262129226383e+00, 0.000000000000000e+00, -1.281164415215970e+00, -5.769506790094477e+00, 0.000000000000000e+00, -5.668871887140390e+00, -5.769506790094458e+00, 0.000000000000000e+00, -5.668871887140382e+00, -7.203316770900582e-04, 0.000000000000000e+00, -6.806874890261611e-04, -3.305305994782313e-03, 0.000000000000000e+00, -3.159391266173417e-03, -2.230538478859623e-03, 0.000000000000000e+00, -2.121101303815518e-03, -1.464606173423320e-03, 0.000000000000000e+00, -1.397767104073322e-03, -1.831068627633592e-03, 0.000000000000000e+00, -1.744283893747324e-03, -1.831068627633592e-03, 0.000000000000000e+00, -1.744283893747324e-03, -2.941090546077626e-04, 0.000000000000000e+00, -2.871027700872251e-04, -4.073898338845556e-02, 0.000000000000000e+00, -4.015807724151918e-02, -3.290871364410483e-02, 0.000000000000000e+00, -3.210828016666044e-02, -1.751724072741424e-02, 0.000000000000000e+00, -1.710709694146509e-02, -2.589281748767859e-02, 0.000000000000000e+00, -2.536237029558946e-02, -2.589281748767859e-02, 0.000000000000000e+00, -2.536237029558946e-02, -2.480760900222847e-03, 0.000000000000000e+00, -2.316503302816689e-03, -2.920245318090485e-01, 0.000000000000000e+00, -2.901847893670261e-01, -2.303463231834232e-01, 0.000000000000000e+00, -2.293612698043842e-01, -1.554265896166024e-01, 0.000000000000000e+00, -1.527175771409462e-01, -1.989609344646014e-01, 0.000000000000000e+00, -2.005885011478925e-01, -1.989609344646015e-01, 0.000000000000000e+00, -2.005885011478926e-01, -5.639958719682883e-02, 0.000000000000000e+00, -5.521807954525737e-02, -1.006448405059849e+00, 0.000000000000000e+00, -1.007958835515458e+00, -1.009060802869413e+00, 0.000000000000000e+00, -1.034090545546599e+00, -1.956646413079852e-01, 0.000000000000000e+00, -1.840303970183562e-01, -1.375333485756338e+00, 0.000000000000000e+00, -1.645785963156414e+00, -1.375333485756339e+00, 0.000000000000000e+00, -1.645785963156415e+00, -1.460044520785298e+00, 0.000000000000000e+00, -1.490735164993722e+00, -7.275859538074726e+00, 0.000000000000000e+00, -1.288828906294180e+01, -4.491032663417388e+00, 0.000000000000000e+00, -4.781131364589934e+00, -1.529599127149264e+00, 0.000000000000000e+00, -1.501049374178628e+00, -1.189621194711300e+01, 0.000000000000000e+00, -5.884458692377099e+00, -1.189621194711301e+01, 0.000000000000000e+00, -5.884458692377115e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05