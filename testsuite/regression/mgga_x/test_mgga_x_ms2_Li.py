
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.952644425506937e+00, -1.368943866350179e+00, -3.729066555421766e-01, -1.757256866376829e-01, -7.621881199680315e-02, -1.712967268006182e-02, -3.200241284136042e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.550686104636876e+00, -2.553065247083230e+00, -1.756066852279818e+00, -1.757864287178737e+00, -3.436535821505579e-01, -4.231898918270263e-01, -2.314705962571277e-01, -2.177864852574420e-02, -8.838210576031470e-02, -6.916761182475200e-04, -2.289804952274230e-02, -2.273321947202948e-02, -4.620011818408396e-04, -3.284407897608138e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.868738599360881e-05, 0.000000000000000e+00, -8.819879743004505e-05, -3.629248304305780e-04, 0.000000000000000e+00, -3.579306567085211e-04, -1.397077528234671e-01, 0.000000000000000e+00, -3.619633671018568e-02, -1.517597009266361e+00, 0.000000000000000e+00, -1.938331803008008e-01, -2.897892952549919e+01, 0.000000000000000e+00, -2.016971560830396e+00, -1.971713316355940e-01, 0.000000000000000e+00, -1.839283541582756e-01, -9.035686113079093e-01, 0.000000000000000e+00, -1.590282313288121e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([9.327913463291070e-06, 9.042164894341583e-12, 6.637644185341292e-05, 1.325491359336548e-17, 2.456017435410247e-02, 7.588289018962268e-11, 6.198375510568338e-03, 9.757097775695735e-18, 6.702799090215554e-07, 3.160665300522458e-10, 9.212924864164493e-22, 3.476644730799265e-18, 4.880566818662295e-39, 1.295246273939376e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
