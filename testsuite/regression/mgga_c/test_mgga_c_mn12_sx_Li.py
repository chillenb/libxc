
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.976999332356414e-01, -1.086861804557150e-01, -2.044425181192019e-02, -2.684848379508842e-02, 4.777987077476829e-03, -1.401226093589840e-01, -3.476903582622115e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.614617855378882e-01, -2.611884464847416e-01, -2.627202350768848e-01, -2.625754184257288e-01, 3.570645708607176e-01, 3.570104125789870e-01, -3.141529446957870e-02, -2.254865610685952e-01, 3.500500390285689e-02, 4.159877701292437e+00, -1.761162011470378e-01, -1.780926035478502e-01, -4.090219255076059e-03, -6.001823714128798e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.446121843551517e-04, -2.892243687103034e-04, -1.446121843551517e-04, -2.338268561605052e-04, -4.676537123210103e-04, -2.338268561605052e-04, 3.828634907246228e-02, 7.657269814492459e-02, 3.828634907246228e-02, -1.045303163756595e+01, -2.090606327513190e+01, -1.045303163756595e+01, 1.298322799773696e+02, 2.596645599547391e+02, 1.298322799773696e+02, 1.515136995313923e-05, 3.030273990696228e-05, 1.515136995313923e-05, 1.406133660571825e-07, 2.812271776566923e-07, 1.406133660571825e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.875836983057559e-02, 2.875836983057556e-02, 3.310503723872344e-02, 3.310503723872343e-02, -9.794811216784674e-02, -9.794811216784671e-02, 5.598764307333973e-01, 5.598764307332750e-01, -7.236455894757288e-01, -7.236455889769130e-01, -2.925367940153313e-07, -2.925367940127048e-07, -7.729234110004035e-19, -7.729379511301825e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
