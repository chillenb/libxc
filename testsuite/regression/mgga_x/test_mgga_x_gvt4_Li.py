
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gvt4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.918589329714595e+00, -1.342181791983931e+00, -4.432677102306765e-01, -1.730161918301601e-01, -8.209913713589218e-02, 1.518898672208433e-05, -5.152880993627944e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gvt4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.518078030366497e+00, -2.520432253813024e+00, -1.704697411729558e+00, -1.706262761804871e+00, -2.077865797875972e-01, -2.126840616587857e-01, -2.292930116411987e-01, -6.799862745921312e-04, -5.175223889187804e-02, -5.539173958230099e-08, 8.770845370985701e-04, -7.531293293296333e-04, 7.138746522284374e-12, -6.054057041209638e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gvt4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.538550531142941e-05, 0.000000000000000e+00, -9.494363694152962e-05, -6.017733022679587e-04, 0.000000000000000e+00, -5.990876365807880e-04, -3.162144012383369e-01, 0.000000000000000e+00, -3.087128246155532e-01, -1.121870989585614e+00, 0.000000000000000e+00, 5.124966859315992e-01, -1.782016942203239e+02, 0.000000000000000e+00, 3.446341780845535e+00, -1.543138369222580e+00, 0.000000000000000e+00, 4.848095952365034e-01, -1.468446850131829e-06, 0.000000000000000e+00, 3.591433567215542e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gvt4_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gvt4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.384108427779690e-03, 2.379766920573585e-03, 4.401126144652333e-03, 4.388657208325944e-03, 4.936357286353484e-02, 4.771603420746456e-02, 2.371880578078179e-02, 6.521970465712590e-05, 1.986323934049740e-01, 1.338424651355186e-08, -1.346233708933799e-06, 7.038473453363094e-05, -1.328192906476808e-17, 1.493231004207085e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
