
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_hf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.104542202517969e-01, -7.929860055989407e-02, 4.013848159447697e-02, 2.914984687019699e-04, -1.461661165880812e-07, -6.853314608597164e-02, -2.064600333197206e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_hf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.459574033767528e-01, -1.452991019505205e-01, -1.773250203188428e-01, -1.767173365222129e-01, 8.277846391995414e-02, 6.772438690731675e-02, 1.864047000802073e-02, -1.540276708678538e+00, 5.964613221219625e-02, -9.271260639716118e-01, -1.192239711184643e-01, -1.073016536230442e-01, -2.881296981199796e-03, -3.584542606643691e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_hf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.118759646628078e-05, 0.000000000000000e+00, -6.157718191516733e-05, 3.886146393479801e-04, 0.000000000000000e+00, 3.862398202062330e-04, -3.390821840145732e-03, 0.000000000000000e+00, -3.551570493757317e-03, -2.152954159077616e+01, 0.000000000000000e+00, 1.035251014742681e+03, -3.480525670790592e+02, 0.000000000000000e+00, 5.233437074862427e+06, 2.492876874174161e+01, 0.000000000000000e+00, 1.144385283803960e+03, 8.051429546041435e+01, 0.000000000000000e+00, 2.475949724052084e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_hf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_hf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.599000514773117e-03, 8.611710059261063e-03, 2.333752349068988e-03, 2.322035817984073e-03, -4.976608224065369e-03, -3.198581450771360e-03, 7.521923119380192e-01, -2.028217371079227e-02, 8.323593825461403e-01, -2.159004626926246e-03, -7.588634558453035e-06, -1.661115042989395e-02, -7.837079475212251e-14, -1.080087038459936e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
