
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbeh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.405781263674849e+00, -1.006160055032575e+00, -3.151180946641863e-01, -1.351808790258522e-01, -6.172558952419799e-02, -1.540837330258630e-02, -2.878940234018740e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbeh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.797970862121440e+00, -1.799445338702890e+00, -1.240699660455868e+00, -1.241631960445837e+00, -3.165001204466618e-01, -3.166383353315719e-01, -1.782016387654696e-01, -1.172273350187842e-01, -6.388275447587406e-02, 3.421963507501030e-01, -2.059298037638717e-02, -2.044500968968799e-02, -4.156166464907097e-04, -2.954656511859193e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbeh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.454084184951780e-04, 9.190971700708733e-05, -1.447492895289730e-04, -6.088071966686765e-04, 2.980993506782570e-04, -6.063755264064657e-04, -5.287882781398719e-02, 6.249948659585063e-03, -5.274060566502103e-02, 4.185276662831661e-01, 6.762268918356340e+00, 3.172847077576420e+00, -3.947902010467219e+01, 2.258698854598489e+01, 9.961154762664435e+00, -2.114964864967252e-01, 3.357174600576258e-04, -1.974893103957890e-01, -9.698924368630927e-01, 3.212885779437900e-06, -1.388301929854175e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
