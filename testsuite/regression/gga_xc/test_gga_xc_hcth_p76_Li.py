
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_p76_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p76", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.785056957580824e+00, -1.189802579457745e+00, -3.595280454413737e-01, -1.003361545130080e-01, -3.853078681785949e-02, -3.900210860536203e-02, -5.342984723304277e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_p76_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p76", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.498228559413345e+00, -2.500672774470165e+00, -1.629520454307771e+00, -1.631190026179060e+00, -1.436651414563571e-01, -1.446853777574735e-01, -1.595359438987652e-01, -1.263194664163735e+00, 5.480178505363260e-03, -7.931167533328088e-01, -5.010124202268830e-02, -5.126067116108080e-02, -3.086500911465748e-04, -1.855058017485864e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_p76_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p76", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.215053792578664e-04, 0.000000000000000e+00, 1.212731014903514e-04, 1.456813690804630e-05, 0.000000000000000e+00, 1.534633481790890e-05, -1.741443577251856e-01, 0.000000000000000e+00, -1.735761792727206e-01, 6.060819918741424e+00, 0.000000000000000e+00, -5.438811023249124e+02, -1.377843189426035e+02, 0.000000000000000e+00, -6.485404015042046e+04, -4.349961147877980e+00, 0.000000000000000e+00, -4.902237666097280e+00, -5.401731872869360e-01, 0.000000000000000e+00, -1.564082927111168e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
