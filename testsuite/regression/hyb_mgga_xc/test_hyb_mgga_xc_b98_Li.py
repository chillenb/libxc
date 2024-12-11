
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b98_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.364573090182809e-01, -1.042420391468050e+00, -3.997945540744015e-01, -1.233027719298141e-01, -6.883414387420989e-02, -1.844233723353857e-02, -3.556296480252319e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b98_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.138426306409952e+00, -1.139209656375862e+00, -1.438745472901003e+00, -1.439827798313529e+00, -3.596906284583564e-01, -3.602292293759148e-01, -2.026911831339334e-01, -1.109406332014742e-02, -5.859882767474245e-02, 6.978073229586340e-03, -2.511945276427074e-02, -2.474939517751691e-02, -5.241036992465109e-04, -3.514164492896394e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b98_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.389430199973578e-04, 0.000000000000000e+00, 4.380888481775582e-04, -5.476443009576968e-04, 0.000000000000000e+00, -5.466336179568374e-04, -6.902439948865070e-02, 0.000000000000000e+00, -6.856989454682003e-02, 2.077732379333104e+01, 0.000000000000000e+00, 1.755565641313870e+01, -7.130418550063342e+01, 0.000000000000000e+00, 7.094609552904395e+04, 3.230121098517448e-01, 0.000000000000000e+00, 1.552504580102981e+01, 1.098559473007479e+00, 0.000000000000000e+00, 3.336760405734497e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b98_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-4.001998055668192e-05, -4.026909134222797e-05, -2.487298543349549e-03, -2.487685972662787e-03, -4.517946251108992e-03, -4.494547124965011e-03, -1.864187085519509e-02, 4.780296140752519e-09, -4.999125518572917e-02, 3.724087965867349e-13, -6.708916345384120e-10, -6.058943413725976e-10, -2.210874785839410e-16, -6.052913789896665e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b98_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.276471406223470e-02, -2.279346458857262e-02, 9.511030763562533e-03, 9.517159892112426e-03, 1.665911859380763e-02, 1.648825978258119e-02, -7.556705158247907e-01, -2.242155670313377e-04, 1.705222629668443e-01, -2.890595106474625e-05, -9.546633249512420e-08, -2.255708885489443e-04, -1.718295138119608e-16, -1.455601221915748e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
