
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt_ge_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.769660130969574e+00, -1.251376528732735e+00, -3.902135203344226e-01, -1.586800571268031e-01, -7.503468491440941e-02, -1.156552160862061e-02, -2.127820881496785e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt_ge_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.270858227120423e+00, -2.272977994837022e+00, -1.549855545619254e+00, -1.551233173652667e+00, -3.475570554215244e-01, -3.475566366638707e-01, -2.069482784469240e-01, -1.601200592342344e-02, -7.304493223051434e-02, -4.598929181543307e-04, -1.755746654131665e-02, -1.708740508895805e-02, -3.071820507310970e-04, -2.183783727950879e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt_ge_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.502222399028499e-04, 0.000000000000000e+00, -1.496927174417703e-04, -6.215063274331098e-04, 0.000000000000000e+00, -6.194511033438413e-04, -8.375729890360371e-02, 0.000000000000000e+00, -8.366789171544131e-02, -2.282384552370396e+00, 0.000000000000000e+00, 1.308535646169980e+01, -5.908519937193855e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.713935699397097e+01, 0.000000000000000e+00, 1.436971446572535e+01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
