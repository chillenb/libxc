
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bkl2_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037953609101132e+01, -2.037957142448178e+01, -2.037978592556893e+01, -2.037920619675508e+01, -2.037955424929841e+01, -2.037955424929841e+01, -3.351771802654091e+00, -3.351758884918152e+00, -3.351542309705572e+00, -3.352626517140983e+00, -3.351779436736908e+00, -3.351779436736908e+00, -6.611183537562774e-01, -6.609126963225285e-01, -6.573908493120370e-01, -6.617303335778284e-01, -6.610408440461499e-01, -6.610408440461499e-01, -1.996215292831560e-01, -2.005631913900654e-01, -7.781787945754506e-01, -1.691840724042996e-01, -1.998783985537223e-01, -1.998783985537223e-01, -1.072511424059150e-02, -1.146176625644218e-02, -6.721623307404574e-02, -4.534576304357000e-03, -1.133762871438729e-02, -1.133762871438729e-02, -4.929182106403916e+00, -4.929298041632208e+00, -4.929198518881241e+00, -4.929288585115741e+00, -4.929234938361759e+00, -4.929234938361759e+00, -1.980609198266433e+00, -1.991040808184225e+00, -1.979970092643764e+00, -1.988066674149840e+00, -1.988602306215260e+00, -1.988602306215260e+00, -5.562785863506523e-01, -5.900007344810104e-01, -5.289987588399364e-01, -5.400812282380287e-01, -5.753085945705103e-01, -5.753085945705103e-01, -1.427474591883693e-01, -2.196336535674046e-01, -1.397661860243132e-01, -1.794694929736913e+00, -1.531724708298892e-01, -1.531724708298892e-01, -4.374318487635419e-03, -5.001645994386427e-03, -3.748094548962222e-03, -9.510428871670819e-02, -4.557445781667301e-03, -4.557445781667301e-03, -5.570947616855640e-01, -5.569036525847011e-01, -5.569725222739430e-01, -5.570235100779966e-01, -5.569975823300251e-01, -5.569975823300251e-01, -5.387697481718744e-01, -4.927865674578127e-01, -5.055528802148527e-01, -5.177623390540625e-01, -5.113956612842436e-01, -5.113956612842436e-01, -6.190974674523213e-01, -2.593266909523167e-01, -2.913430801435872e-01, -3.474899240606333e-01, -3.167273804053272e-01, -3.167273804053271e-01, -4.462012360309958e-01, -6.129683895084637e-02, -8.554812745071189e-02, -3.224754484621072e-01, -1.188271816546616e-01, -1.188271816546616e-01, -1.305394310082809e-02, -1.368592183343880e-03, -2.599826628724452e-03, -1.132201411097145e-01, -3.871716868249165e-03, -3.871716868249159e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bkl2_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.571387408737997e+01, -2.571395388993775e+01, -2.571433073136599e+01, -2.571302351405894e+01, -2.571391599847203e+01, -2.571391599847203e+01, -4.145353988624875e+00, -4.145386441503049e+00, -4.146448673575421e+00, -4.145496553353783e+00, -4.145388338926015e+00, -4.145388338926015e+00, -7.735213284454872e-01, -7.722203419608241e-01, -7.372498068958833e-01, -7.438720466937295e-01, -7.730493386835338e-01, -7.730493386835338e-01, -1.958734443303331e-01, -1.995335299283512e-01, -9.428059109324806e-01, -1.411846764196010e-01, -1.969782097082324e-01, -1.969782097082324e-01, -2.022793863136665e-02, -2.189488873568864e-02, -9.915680397173827e-02, -6.055997795126884e-03, -2.165308665707891e-02, -2.165308665707891e-02, -6.310806038794813e+00, -6.312885800791537e+00, -6.311020189733501e+00, -6.312637315210525e+00, -6.311870725978617e+00, -6.311870725978617e+00, -2.225685763372251e+00, -2.244834921587654e+00, -2.214472033040453e+00, -2.229427456388882e+00, -2.253177931194799e+00, -2.253177931194799e+00, -6.955071013726656e-01, -7.734235164309422e-01, -6.585828142706928e-01, -7.014563633702023e-01, -7.250888678203895e-01, -7.250888678203895e-01, -1.347243577880464e-01, -1.915821406046990e-01, -1.294870950746856e-01, -2.352341307977972e+00, -1.294695879349219e-01, -1.294695879349219e-01, -5.839153508174447e-03, -6.723163675947492e-03, -5.026820444288826e-03, -1.211498140968282e-01, -6.128890680235496e-03, -6.128890680235496e-03, -7.391355783381088e-01, -7.315290477721967e-01, -7.342495044614469e-01, -7.363556236682044e-01, -7.353015576490385e-01, -7.353015576490385e-01, -7.160047754315884e-01, -5.899654085439973e-01, -6.268969396925366e-01, -6.612226252994171e-01, -6.438551567536916e-01, -6.438551567536916e-01, -8.103063019199865e-01, -2.447987036820237e-01, -3.007256113227943e-01, -4.074154351524894e-01, -3.508416158244676e-01, -3.508416158244675e-01, -5.301729290019704e-01, -9.316450289252137e-02, -1.184843081814230e-01, -3.914635950702413e-01, -1.203670713630746e-01, -1.203670713630745e-01, -2.512502801298045e-02, -1.824789577791840e-03, -3.466435539159386e-03, -1.142157680707627e-01, -5.180218479866192e-03, -5.180218479866184e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bkl2_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.693809263851596e-09, -3.693777094489885e-09, -3.693601830445499e-09, -3.694129261114068e-09, -3.693792564224159e-09, -3.693792564224159e-09, -5.180322866838045e-06, -5.180324992122202e-06, -5.179529122958766e-06, -5.176508571723276e-06, -5.180235690794721e-06, -5.180235690794721e-06, -3.672306702811873e-03, -3.683220508045094e-03, -3.959903175115281e-03, -3.845798419323681e-03, -3.676305427288554e-03, -3.676305427288554e-03, -5.471894788165900e-01, -5.282973718563020e-01, -1.830530927551542e-03, -1.268791883632588e+00, -5.416035647019524e-01, -5.416035647019524e-01, 2.646623662465752e+02, 2.573902160209040e+02, 3.799461639353089e+00, 3.352605405923828e+00, 2.675894838314282e+02, 2.675894838314282e+02, -1.059059057935942e-06, -1.058537260590645e-06, -1.059002719915877e-06, -1.058597039728199e-06, -1.058795667327121e-06, -1.058795667327121e-06, -4.794062137561615e-05, -4.675170639749867e-05, -4.828377384090784e-05, -4.734681361732004e-05, -4.669498865191213e-05, -4.669498865191213e-05, -6.732965945637283e-03, -5.002415525376848e-03, -8.278401252533003e-03, -7.211713457460927e-03, -5.824697740002848e-03, -5.824697740002848e-03, -1.777926543838134e+00, -4.275954465972467e-01, -2.048958730145115e+00, -5.843862327323346e-05, -1.806212400783099e+00, -1.806212400783099e+00, 2.615227315115177e+00, 1.415865600654215e+01, 3.075894935333663e+01, -8.776637373610893e-01, 2.190788752377581e+01, 2.190788752377574e+01, -6.194013461005289e-03, -6.284955016669687e-03, -6.252200792585552e-03, -6.227076959651139e-03, -6.239645448820485e-03, -6.239645448820485e-03, -7.065222411827382e-03, -1.155672700632689e-02, -9.975165951235661e-03, -8.728124662255919e-03, -9.342068539529211e-03, -9.342068539529210e-03, -4.134676097343714e-03, -2.010286334931824e-01, -1.135532896118493e-01, -4.799028431764350e-02, -7.461924002907441e-02, -7.461924002907443e-02, -1.735799877445465e-02, 5.199428818854351e+00, 8.462660776477421e-01, -6.191881650519326e-02, -2.859690829669056e+00, -2.859690829669060e+00, 1.857813055238685e+02, 3.943940674422046e-26, 7.705354073761883e-05, -3.529364601556314e+00, 1.478432987567556e+01, 1.478432987567520e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05