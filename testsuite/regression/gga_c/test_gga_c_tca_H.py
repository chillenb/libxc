
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tca_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.636834384146470e-02, -1.919075517230688e-02, -9.062551660286248e-03, -3.077643805053817e-04, -1.160729419206103e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tca_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.986160791777352e-02, -1.635439124093735e+03, -3.605194300568272e-02, -1.067626180327526e+03, -2.573625784644216e-02, -2.947552203740242e+02, -1.298573523958525e-03, -2.673801481114392e+00, -5.102822952728016e-08, -4.777207258442910e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tca_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.810368193593808e-03, 7.620736387187616e-03, 3.810368193593808e-03, 6.657912885806471e-03, 1.331582577161294e-02, 6.657912885806471e-03, 3.292780234684729e-02, 6.585560469369459e-02, 3.292780234684729e-02, 1.021866129635035e-01, 2.043732259270070e-01, 1.021866129635035e-01, 2.844304410913167e-02, 5.688608821826333e-02, 2.844304410913167e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
