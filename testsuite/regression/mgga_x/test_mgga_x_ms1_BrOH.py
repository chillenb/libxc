
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms1_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.260332925462665e+01, -2.260337721231998e+01, -2.260364910060587e+01, -2.260286319765957e+01, -2.260335405418068e+01, -2.260335405418068e+01, -3.371185856684131e+00, -3.371200212890323e+00, -3.371869968740425e+00, -3.373095038151006e+00, -3.371198200274260e+00, -3.371198200274260e+00, -6.414370231954957e-01, -6.411624353231120e-01, -6.364511720262337e-01, -6.452024415437507e-01, -6.413406560599390e-01, -6.413406560599390e-01, -1.993753961343232e-01, -2.005344732536002e-01, -7.221635795732264e-01, -1.506472524054649e-01, -1.996170278466040e-01, -1.996170278466040e-01, -1.325033561333269e-02, -1.387708982801330e-02, -5.549386683909580e-02, -6.364162506482740e-03, -1.373251899190837e-02, -1.373251899190837e-02, -5.489656744297415e+00, -5.490223467367491e+00, -5.489719001793429e+00, -5.490159581930206e+00, -5.489941252295492e+00, -5.489941252295492e+00, -2.103486234455606e+00, -2.123049504478351e+00, -2.103452265146325e+00, -2.118567164478449e+00, -2.117033649176455e+00, -2.117033649176455e+00, -5.914528978714833e-01, -6.346746498317719e-01, -5.408114250846224e-01, -5.559187347602577e-01, -6.224238436985053e-01, -6.224238436985053e-01, -1.177480424396818e-01, -2.080599773783941e-01, -1.158493338802602e-01, -1.838062092621696e+00, -1.325288996088921e-01, -1.325288996088921e-01, -6.139497163864751e-03, -7.015854840882148e-03, -5.258511819581398e-03, -7.600206894052124e-02, -6.392511118947608e-03, -6.392511118947608e-03, -6.257320655925668e-01, -6.239771524822478e-01, -6.246088144893983e-01, -6.250931995294429e-01, -6.248503984307431e-01, -6.248503984307431e-01, -6.043013519871345e-01, -5.384290156562923e-01, -5.575049266430365e-01, -5.751914037693674e-01, -5.660225528944461e-01, -5.660225528944461e-01, -6.523662556146286e-01, -2.592904398577869e-01, -2.964890506387011e-01, -3.538198904439118e-01, -3.262357645001454e-01, -3.262357645001454e-01, -4.728363758724827e-01, -5.159772077955422e-02, -6.899989227090449e-02, -3.301985524738189e-01, -9.662789464807854e-02, -9.662789464807854e-02, -1.552492054344757e-02, -1.921436262717880e-03, -3.649729301375092e-03, -9.212400188955538e-02, -5.433036687530268e-03, -5.433036687530261e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms1_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.919382139614004e+01, -2.919391199179658e+01, -2.919443849952519e+01, -2.919296003908345e+01, -2.919386805413681e+01, -2.919386805413681e+01, -4.255933093257481e+00, -4.256193797358976e+00, -4.264453139802360e+00, -4.264210512282303e+00, -4.256008901147941e+00, -4.256008901147941e+00, -7.953790655546433e-01, -7.936386591579778e-01, -7.517009887780237e-01, -7.670708931345467e-01, -7.947597168130612e-01, -7.947597168130612e-01, -1.862186274954815e-01, -1.903524197498316e-01, -8.754814699160204e-01, -1.619526444600116e-01, -1.873987360956877e-01, -1.873987360956877e-01, -1.761525668618913e-02, -1.844312137627349e-02, -7.141506184622552e-02, -8.480733862016870e-03, -1.825108384156929e-02, -1.825108384156929e-02, -7.176302276049544e+00, -7.178081984855671e+00, -7.176518222608880e+00, -7.177899977375783e+00, -7.177169291512145e+00, -7.177169291512145e+00, -2.359581318857332e+00, -2.420199745932742e+00, -2.365449844979225e+00, -2.416454166518626e+00, -2.394733450197756e+00, -2.394733450197756e+00, -7.654225436114085e-01, -8.954226297552432e-01, -7.097930235231628e-01, -7.936893776647715e-01, -8.038461939426184e-01, -8.038461939426184e-01, -1.370497654039101e-01, -1.865560277720273e-01, -1.340508833780170e-01, -2.665142766210539e+00, -1.448334453596784e-01, -1.448334453596784e-01, -8.181542880388226e-03, -9.347819066727868e-03, -7.006609293992699e-03, -9.455362061445599e-02, -8.517230843801796e-03, -8.517230843801796e-03, -8.327506774875956e-01, -8.249039379391183e-01, -8.276896397680881e-01, -8.298619012035896e-01, -8.287729531561128e-01, -8.287729531561128e-01, -8.083091660276891e-01, -6.711128128539359e-01, -7.076652920749489e-01, -7.447331813858435e-01, -7.251718470783303e-01, -7.251718470783303e-01, -9.402374964332642e-01, -2.529855770004249e-01, -2.905394165083912e-01, -4.211317491016362e-01, -3.486250733979986e-01, -3.486250733979986e-01, -5.585219603230182e-01, -6.690347371165613e-02, -8.736585375852016e-02, -4.147220842595422e-01, -1.144774883390502e-01, -1.144774883390502e-01, -2.062757230944224e-02, -2.561735928415858e-03, -4.865166565177398e-03, -1.090969601688873e-01, -7.239511035764781e-03, -7.239511035764770e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.262321745059505e-09, -3.262242126279626e-09, -3.261175082508958e-09, -3.262435347882551e-09, -3.262286448036270e-09, -3.262286448036270e-09, -8.411016504344738e-06, -8.419319265578369e-06, -8.680543418609915e-06, -8.705869305714515e-06, -8.412347694027580e-06, -8.412347694027580e-06, -6.450569405411470e-03, -6.445997581646866e-03, -6.260439853189310e-03, -6.335093396328018e-03, -6.449337222303920e-03, -6.449337222303920e-03, -1.465354530344837e+00, -1.430347471174612e+00, -2.749334558829884e-03, -6.610991474920875e-01, -1.454974809873125e+00, -1.454974809873125e+00, -2.317450180653959e+00, -2.323832029497763e+00, -1.031397356687878e+00, -1.631623674842194e+00, -2.415244525817117e+00, -2.415244525817117e+00, -5.969545763198998e-07, -5.965950589132753e-07, -5.965775419614826e-07, -5.963185567034203e-07, -5.972280729268617e-07, -5.972280729268617e-07, -1.187298540384312e-04, -1.024532514836983e-04, -1.113155582913984e-04, -9.772919443212514e-05, -1.168662630823773e-04, -1.168662630823773e-04, -2.745730059400831e-02, -2.174196452844680e-02, -2.750626382382754e-02, -2.809491663651843e-02, -2.123287306041299e-02, -2.123287306041299e-02, -6.960816319416678e-01, -8.320813056271082e-01, -8.127118451154581e-01, -2.167962608839441e-04, -1.133207951890827e+00, -1.133207951890827e+00, -1.730827360243235e+00, -1.734961161128923e+00, -4.965464766947557e+00, -1.054788564081473e+00, -2.562656109948204e+00, -2.562656109948203e+00, -6.688626223272944e-03, -6.616501902941940e-03, -6.643490777652114e-03, -6.663202074059065e-03, -6.654959675858158e-03, -6.654959675858156e-03, -1.145913140962585e-02, -1.265117014334807e-02, -1.287104153672914e-02, -1.262961836873939e-02, -1.323325756382481e-02, -1.323325756382482e-02, -1.862703428058508e-02, -3.945802440326169e-01, -3.328134168485826e-01, -1.437161449002616e-01, -2.491730452297900e-01, -2.491730452297901e-01, -6.252542470156595e-02, -8.710710444965863e-01, -8.938661998810358e-01, -2.071494204221531e-01, -1.185983122864193e+00, -1.185983122864193e+00, -1.740470408686701e+00, -2.980452009080467e+00, -2.567072941964374e+00, -1.428451790266831e+00, -3.741888739741037e+00, -3.741888739741032e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms1_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.019705582613106e-04, 2.019618664911053e-04, 2.018184300633019e-04, 2.019540086731113e-04, 2.019669628869991e-04, 2.019669628869991e-04, 2.157277462422758e-03, 2.162881691410395e-03, 2.340542028106516e-03, 2.364669152571921e-03, 2.158236617863232e-03, 2.158236617863232e-03, 1.213182726717714e-02, 1.206186759017262e-02, 1.045562229605254e-02, 1.201736336057941e-02, 1.210872820796551e-02, 1.210872820796551e-02, 1.146542917978135e-01, 1.151656322256535e-01, 2.389418742865982e-03, 3.017778653481885e-03, 1.146854964107840e-01, 1.146854964107840e-01, 5.601219153533006e-08, 4.187507326095445e-08, 8.329756272156962e-06, 3.060672880145382e-15, 6.489999054751684e-08, 6.489999054752379e-08, 5.938979715081823e-05, 5.871529764721847e-05, 5.856491451396024e-05, 5.808604236665868e-05, 6.006086594109815e-05, 6.006086594109815e-05, 1.119604601067464e-02, 9.614012285433945e-03, 1.027870852518086e-02, 8.939605610386970e-03, 1.129866418776081e-02, 1.129866418776081e-02, 6.975704489640182e-02, 7.093466744822163e-02, 5.245061567583944e-02, 6.300347730259802e-02, 6.034204388034035e-02, 6.034204388034035e-02, 1.091715311401352e-03, 6.469116849214725e-02, 1.389721104411803e-03, 1.747686923084242e-02, 1.107794240006765e-02, 1.107794240006765e-02, 2.675869479587919e-14, 2.193388619298173e-14, 3.586230220838058e-13, 9.128404886886254e-06, 9.389152821609012e-15, 9.389152820911801e-15, 1.079057172868137e-02, 1.036971041049941e-02, 1.052322303432535e-02, 1.063867550853256e-02, 1.058617990051905e-02, 1.058617990051905e-02, 2.274588703403348e-02, 1.394887227114592e-02, 1.759197488847630e-02, 2.029008389289283e-02, 2.002771396052081e-02, 2.002771396052085e-02, 6.857529754615412e-02, 6.183971488323595e-02, 8.930646381489332e-02, 7.147025153296258e-02, 9.564488476652624e-02, 9.564488476652626e-02, 7.602365469719269e-02, 1.537234387870653e-05, 2.276907989465555e-05, 8.859534851014005e-02, 1.118865428918839e-03, 1.118865428918838e-03, 1.319641665355891e-11, 9.025810727038366e-20, 7.186257528898358e-15, 9.751569798147753e-04, 4.671024052200764e-15, 4.671024066965842e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05