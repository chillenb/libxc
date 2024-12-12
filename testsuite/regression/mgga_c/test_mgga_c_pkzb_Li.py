
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_pkzb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.913384938201482e-02, -4.615581024209597e-02, -4.045086783048824e-03, -1.465112300347046e-03, 1.413368993577470e-07, -6.850792370281822e-09, -1.701772134556342e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_pkzb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.926109042472850e-02, -9.908509605559129e-02, -8.698188375487252e-02, -8.675837875536116e-02, -1.869853051553142e-02, -1.858134012798629e-02, -3.051036577768451e-02, 1.976582543432299e+00, -2.696112860794497e-03, 5.486467134081220e-01, -4.927571825156655e-08, -4.264951132949074e-08, -1.135776295461281e-15, -1.213434990348613e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.016445292089294e-06, 1.405973226208258e-04, -1.293757161947543e-06, -7.126946124263543e-05, 4.558836461192621e-04, -6.931508342239037e-05, 8.004858719493034e-04, 9.469816248209570e-03, 8.025888988300896e-04, 3.681543768412858e+01, 1.002517605820849e+01, -5.432703376612790e+04, 1.573441522030819e+01, 3.455808780051290e+01, -4.894515761431389e+07, 1.677171165267824e-04, 3.360066927921953e-04, 1.677300533772196e-04, 1.606719760827460e-06, 3.213444612894168e-06, 1.777344821578671e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.854716364857263e-04, -4.251973857366185e-04, 1.335672907102346e-03, 1.292350043074884e-03, 6.066064637071535e-04, 5.782121033228584e-04, -1.331300621058717e+00, 6.620990348750343e-01, -3.763052831942710e-02, 1.994199372698151e-02, 1.617453925830835e-13, -9.859127339002607e-10, 7.875913869690310e-32, -5.649260558487408e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
