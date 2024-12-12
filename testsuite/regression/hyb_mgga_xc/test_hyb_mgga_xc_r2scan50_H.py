
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan50_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.648925707615376e-01, -3.270283845763932e-01, -1.907776216075268e-01, -4.849822053897693e-02, -1.085448592047901e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan50_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.864452159796401e-01, -2.199798770977095e-01, -4.322158336928570e-01, -2.101325999806031e-01, -2.511381263075008e-01, -1.813309671284328e-01, -6.901421506667574e-02, -1.091393373285240e-01, -3.968277442505423e-02, -9.891645420874053e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan50_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.199199028321679e-03, 8.472540629547595e-03, 4.236270314773800e-03, -5.333252068580645e-03, 9.771615395811520e-03, 4.885807697905760e-03, -2.708193345010432e-02, 9.485995032994637e-02, 4.742997516497319e-02, 3.353291064240587e-01, 2.306501582130026e+01, 1.153250791065013e+01, 8.077110193631734e+04, 4.144222864515410e+05, 2.072111432257705e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan50_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([7.630169261945751e-03, -9.984991666357742e-03, 9.787476185837677e-03, -8.263185613481468e-03, 1.102154215995082e-02, -1.625762109617745e-02, 1.617626489536936e-03, -7.556336542252602e-02, -5.504683611090933e-02, -1.419598229148764e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
