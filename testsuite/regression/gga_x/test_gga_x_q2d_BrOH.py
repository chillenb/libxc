
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_q2d_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.046766101165359e+01, -2.046769499416380e+01, -2.046790588855933e+01, -2.046734823153734e+01, -2.046767843726167e+01, -2.046767843726167e+01, -3.369134599668776e+00, -3.369120398147865e+00, -3.368869064006519e+00, -3.370016616566519e+00, -3.369141647489485e+00, -3.369141647489485e+00, -6.641877511804442e-01, -6.639276273374433e-01, -6.574807900643614e-01, -6.620681324561358e-01, -6.640909386458481e-01, -6.640909386458481e-01, -1.871449205777414e-01, -1.904160523277144e-01, -7.824900678728387e-01, -8.091945042362654e-02, -1.881723951685893e-01, -1.881723951685893e-01, -9.556149143675773e-04, -1.023009331232618e-03, -6.700891766519607e-03, -3.093621307119499e-04, -1.011903830037836e-03, -1.011903830037836e-03, -4.946097226253042e+00, -4.946110529322474e+00, -4.946103409418697e+00, -4.946113671378325e+00, -4.946097099330928e+00, -4.946097099330928e+00, -1.981512564170370e+00, -1.992945033097721e+00, -1.979350356743031e+00, -1.988317492093178e+00, -1.991872603174342e+00, -1.991872603174342e+00, -5.589327165712892e-01, -5.909350113362749e-01, -5.316164521745625e-01, -5.413573863501216e-01, -5.778304204265531e-01, -5.778304204265531e-01, -2.364598667600484e-02, -1.579374277939037e-01, -2.447444508353353e-02, -1.797556810761464e+00, -4.457296547602272e-02, -4.457296547602272e-02, -2.953597267099881e-04, -3.603849312442497e-04, -2.667804531100532e-04, -1.045881246997669e-02, -3.290824144928870e-04, -3.290824144928871e-04, -5.573612380776650e-01, -5.576850511740824e-01, -5.575734550354265e-01, -5.574816389527553e-01, -5.575274458088547e-01, -5.575274458088547e-01, -5.389419026656773e-01, -4.954807973804177e-01, -5.081282658404532e-01, -5.196263971362891e-01, -5.136655272509529e-01, -5.136655272509529e-01, -6.201615409518131e-01, -2.315910569295525e-01, -2.836277615086686e-01, -3.491433104483859e-01, -3.160644353502929e-01, -3.160644353502929e-01, -4.485666638070740e-01, -5.969477170273558e-03, -8.912384726939091e-03, -3.242591806310696e-01, -1.680318948329899e-02, -1.680318948329902e-02, -1.166029917382786e-03, -5.564231153651225e-05, -1.426604951197831e-04, -1.613049325074116e-02, -2.705523711575874e-04, -2.705523711575868e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_q2d_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.566986198863179e+01, -2.566993953623625e+01, -2.567030934672819e+01, -2.566903901070356e+01, -2.566990268520397e+01, -2.566990268520397e+01, -4.147091485547637e+00, -4.147116338058570e+00, -4.147971087037906e+00, -4.147378460276570e+00, -4.147121933741630e+00, -4.147121933741630e+00, -7.890397319944220e-01, -7.883577275112540e-01, -7.794293552556686e-01, -7.843588092599196e-01, -7.887884884123633e-01, -7.887884884123633e-01, -2.880795906931118e-01, -2.784057937486563e-01, -9.474652215861080e-01, -3.587887396323206e-01, -2.848218158382353e-01, -2.848218158382353e-01, -1.862575103679809e-03, -1.990182568022784e-03, -1.161793888888448e-02, -6.140509361649454e-04, -1.968655282151694e-03, -1.968655282151694e-03, -6.297125734455172e+00, -6.299193760229157e+00, -6.297338606834397e+00, -6.298946588812078e+00, -6.298184128485230e+00, -6.298184128485230e+00, -2.347728605736223e+00, -2.359478062743384e+00, -2.348524816250131e+00, -2.357153743389573e+00, -2.356377358009753e+00, -2.356377358009753e+00, -6.948262794749924e-01, -7.723347061883349e-01, -6.582519913569860e-01, -7.001257052950314e-01, -7.238981927414394e-01, -7.238981927414394e-01, -8.353046144364761e-02, -4.780548138088321e-01, -9.128198313907131e-02, -2.349010493406962e+00, -2.133151350419080e-01, -2.133151350419080e-01, -5.863955314056025e-04, -7.143351177843225e-04, -5.289736941253979e-04, -1.941844018729685e-02, -6.522499650181881e-04, -6.522499650181888e-04, -7.387902359871329e-01, -7.305993073428645e-01, -7.335100042895897e-01, -7.357783323702773e-01, -7.346415743762110e-01, -7.346415743762110e-01, -7.157791049159649e-01, -5.952208039891299e-01, -6.269122462702050e-01, -6.598036667224455e-01, -6.428436067930623e-01, -6.428436067930623e-01, -8.090866625593767e-01, -4.256165276792078e-01, -3.724784638437446e-01, -4.150897816666967e-01, -3.766066665478176e-01, -3.766066665478175e-01, -5.365488429025342e-01, -1.050184335618152e-02, -1.549045356762322e-02, -3.931826694199910e-01, -4.870607334021181e-02, -4.870607334021195e-02, -2.264525519300736e-03, -1.111068805938441e-04, -2.842031068099504e-04, -4.727160783564263e-02, -5.367078505632881e-04, -5.367078505632869e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_q2d_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.102760354589135e-09, -4.102734936737629e-09, -4.102571360793527e-09, -4.102988576148366e-09, -4.102747369573012e-09, -4.102747369573012e-09, -5.523029912109704e-06, -5.523179044224665e-06, -5.526357326044739e-06, -5.516189577216090e-06, -5.523012507516350e-06, -5.523012507516350e-06, -3.283685916771058e-03, -3.273745833726927e-03, -2.764021314352849e-03, -2.733567310032922e-03, -3.280221231970896e-03, -3.280221231970896e-03, 3.001313717556447e-01, 1.908054617000783e-01, -1.851567485455776e-03, 3.772005800480903e+00, 2.642748383449333e-01, 2.642748383449333e-01, 2.627171766148768e+01, 2.437342034780190e+01, 1.069257182608198e+00, 6.828763769723898e+01, 2.535986138982268e+01, 2.535986138982268e+01, -1.205839219936680e-06, -1.205823148465460e-06, -1.205832987878490e-06, -1.205820591879987e-06, -1.205838042233567e-06, -1.205838042233567e-06, -3.398568044402421e-05, -3.396507923180828e-05, -3.297788867754234e-05, -3.303113263964994e-05, -3.510728510388605e-05, -3.510728510388605e-05, -7.347950108835924e-03, -5.884179979402187e-03, -8.955047407467588e-03, -8.384104773251211e-03, -6.456299440476984e-03, -6.456299440476984e-03, 1.662721172366345e+00, 1.129434564801497e+00, 2.113149050300209e+00, -6.872907558204975e-05, 3.717853542180927e+00, 3.717853542180927e+00, 7.484883441068337e+01, 6.096611849821983e+01, 1.815157811042675e+02, 8.493366762100549e-01, 8.942795358141271e+01, 8.942795358141269e+01, -7.380581913805894e-03, -7.410513638497861e-03, -7.401656187269610e-03, -7.393605028260330e-03, -7.397810070170777e-03, -7.397810070170777e-03, -8.430959414723055e-03, -1.127021977693867e-02, -1.069803926778928e-02, -9.898091879068300e-03, -1.033464602673168e-02, -1.033464602673168e-02, -4.854406564378491e-03, 2.326018956299802e-01, -7.367135227679512e-03, -4.329479591042781e-02, -4.679272042676443e-02, -4.679272042676440e-02, -1.649441204366576e-02, 1.156039693715396e+00, 6.906375752295478e-01, -6.297895088752570e-02, 1.975747666197688e+00, 1.975747666197701e+00, 1.708225559561750e+01, 6.143873810607247e+02, 2.118006184869436e+02, 2.474703641726458e+00, 1.451041228455565e+02, 1.451041228455578e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05