
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_20_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.178276477228257e+00, -1.524136379945070e+00, -2.927385116042332e-01, -1.949969209442160e-01, -6.815629052316974e-02, -1.077824120699416e-02, -2.028559451334642e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_20_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.664161211684557e+00, -2.666735481279225e+00, -1.792939617186534e+00, -1.794281182484980e+00, -3.849797434738857e-01, -3.868914278677048e-01, -2.467005132061185e-01, -1.331615557021980e-02, -8.998206792951399e-02, -4.222849581140265e-04, -1.460395482540603e-02, -1.390108490441107e-02, -2.943206532308657e-04, -2.005203418539180e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_20_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.462596272374586e-04, 0.000000000000000e+00, -8.434244559762485e-04, -3.263319187250679e-03, 0.000000000000000e+00, -3.254957173256679e-03, -5.301990661106494e-02, 0.000000000000000e+00, -5.625758878309793e-02, -1.300554390911452e+01, 0.000000000000000e+00, -6.154270755122951e+00, -8.906655244685281e+01, 0.000000000000000e+00, -1.530777855409688e+04, 3.699318906392469e-01, 0.000000000000000e+00, -5.504988024685737e+00, 7.790157909832923e-01, 0.000000000000000e+00, -6.929971464957956e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_20_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.625455718023374e-02, 3.621680687381496e-02, 5.573462802356222e-02, 5.571079388318293e-02, 1.908339759270557e-02, 2.080452305577595e-02, 3.798283337688566e-01, 7.899610796460225e-05, 3.333237135321692e-01, 6.237011431191482e-06, -1.125495633330031e-07, 8.041179621214504e-05, -7.489592892217848e-16, 3.023083371933442e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
