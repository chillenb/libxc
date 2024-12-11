
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.307784514985736e-01, -1.060614989095946e-01, -3.370895204948132e-01, -4.391238507902791e-02, -5.924924859549317e-02, -4.808805093045721e-02, -1.193367781547953e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.243221714664421e-01, -6.239225134965207e-01, -2.807234503919487e-01, -2.805477162564131e-01, 2.937539184473643e-02, 2.907953856293240e-02, -1.216051539021391e-01, -3.724006828315892e-01, 1.677199935023960e-02, -2.990838360292787e-01, -6.043080298453243e-02, -6.110907567308781e-02, -1.403874384874270e-03, -2.059988976000360e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.065819973134881e-05, -6.131639946269763e-05, -3.065819973134881e-05, -3.767401512716343e-04, -7.534803025432686e-04, -3.767401512716343e-04, 1.766315824950405e+00, 3.532631649900811e+00, 1.766315824950405e+00, -9.552253450052762e-01, -1.910450690010552e+00, -9.552253450052762e-01, 1.424465881434008e+03, 2.848931762868016e+03, 1.424465881434008e+03, 9.415167043721570e-08, 1.883033459717190e-07, 9.415167043721570e-08, 9.915807928565500e-16, -1.754007597259695e-14, 9.915807928565500e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.276040737922703e-02, 5.276040737922701e-02, 2.390795499298907e-02, 2.390795499298906e-02, -7.622349655364519e-02, -7.622349655364512e-02, 1.916181149032131e+00, 1.916181149031713e+00, -7.336487668105957e-01, -7.336487663048842e-01, -1.766800566855286e-07, -1.766800566848041e-07, -4.668867983400107e-19, -4.668867983400107e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
