
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.503471246644000e+00, -1.085912763025200e+00, -3.357997196566219e-01, -1.348027676660297e-01, -6.812051401017361e-02, -1.011621654798785e-01, -3.875130061848256e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.875140110185602e+00, -1.876658388836631e+00, -1.300505617705704e+00, -1.301425295030610e+00, -3.744389844827844e-01, -3.746886071592009e-01, -1.717905178928183e-01, -1.173806095595172e-01, -6.473005102937207e-02, -4.498981878312285e-02, -3.301467320572870e-02, -3.322084042430754e-02, -5.559402200252651e-03, -4.865431191099777e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.919101935266067e-04, 4.230480491699886e-06, -1.913582478680785e-04, -7.091371939437872e-04, 2.954022849291356e-05, -7.073439109924753e-04, -5.453687912656912e-02, 3.866747919504811e-02, -5.438428174813650e-02, -3.175110157836661e+00, 3.722869163256963e+00, -9.617113315518748e+02, -5.514852799530517e+01, 1.909121184807025e+01, -3.492216945794654e+07, -8.386710703408741e+02, 6.428238830639903e-02, -8.400092928335994e+02, -1.036801610330859e+08, 0.000000000000000e+00, -3.088520081156214e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
