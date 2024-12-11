
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.126423071283970e-02, -5.298284383172547e-02, -1.462477590399576e-01, -7.516672677906811e-03, -2.370765459537410e-02, -5.774965859706187e-02, -1.433180880271527e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.823422491463651e-01, -2.822708548643880e-01, 8.787201317481536e-02, 8.795980086737777e-02, 1.065002951209286e-01, 1.063719378839395e-01, -5.094658154540092e-02, -9.387683547238868e-02, 1.431626399487873e-02, -1.120684788986092e-01, -7.256905239923905e-02, -7.338360020165074e-02, -1.685989816213779e-03, -2.473953846924613e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.864969740900502e-04, -7.729939481801003e-04, -3.864969740900502e-04, -1.002483076238833e-03, -2.004966152477666e-03, -1.002483076238833e-03, 5.891125442900943e-01, 1.178225088580189e+00, 5.891125442900943e-01, -1.217562642258718e+01, -2.435125284517437e+01, -1.217562642258718e+01, 4.685411014663774e+02, 9.370822029327547e+02, 4.685411014663774e+02, 1.208958109314002e-07, 2.417916284079920e-07, 1.208958109314002e-07, 1.273466077935569e-15, -2.252634572637022e-14, 1.273466077935569e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.373473642069297e-02, 3.373473642069294e-02, -2.141726392895485e-02, -2.141726392895485e-02, -5.031663503816223e-02, -5.031663503816217e-02, 1.146051903749424e+00, 1.146051903749174e+00, -3.572360444403956e-01, -3.572360441941493e-01, -2.372866033584794e-07, -2.372866033657249e-07, -6.268023767881646e-19, -6.268605373072802e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
