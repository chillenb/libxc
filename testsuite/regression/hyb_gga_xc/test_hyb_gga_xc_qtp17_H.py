
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_qtp17_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_qtp17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.468019585165826e-01, -2.220831404734804e-01, -1.327026749623833e-01, -3.867609063723526e-02, -2.664385417462042e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_qtp17_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_qtp17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.264762320073865e-01, -2.370115673344244e-01, -2.936120612921587e-01, -2.460611720331384e-01, -1.748823000601258e-01, -1.944414753692451e-01, -5.043702612183422e-02, -4.761170343811291e-02, -3.433078373729746e-03, -3.716508185570551e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_qtp17_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_qtp17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.009040637487447e-16, 2.339856093867776e-02, 1.754219946320068e-02, -1.280348838626414e-15, 3.717534120472241e-02, 2.784532062649763e-02, -4.397372752249522e-14, 3.189417211668060e-01, 2.391992236940503e-01, 2.749251017868760e-11, 1.362229630262558e+01, 1.021670494113630e+01, 1.235997581997870e-23, 5.726247898060341e-18, 4.294679500260225e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
