
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_vcml_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.837450226924592e+00, -1.246418056710312e+00, -2.823823792636329e-01, -1.648717430074303e-01, -6.096516779723388e-02, -1.139347148644794e-02, -2.112035211704329e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_vcml_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.414397734862857e+00, -2.416328000683913e+00, -1.869062545018601e+00, -1.870542455982624e+00, -3.777365156710282e-01, -3.776554748361297e-01, -2.160986499548712e-01, -1.467591799512432e-02, -8.232198313287312e-02, -4.585853169658289e-04, -1.521066518462952e-02, -1.533420105228328e-02, 7.881348869379073e+02, -2.128252775768312e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vcml_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.434243855633943e-05, 0.000000000000000e+00, -6.352424531396871e-05, -2.092336960692883e-03, 0.000000000000000e+00, -2.082478169081304e-03, -2.469913274815860e-01, 0.000000000000000e+00, -2.473931843024312e-01, -9.258813121628526e-01, 0.000000000000000e+00, 1.380361555636208e+00, -1.161422180061402e+02, 0.000000000000000e+00, -4.989875929910592e+01, 5.963359865877868e-04, 0.000000000000000e+00, 1.308805890896511e+00, -3.623652983495957e+07, 0.000000000000000e+00, -2.417021697575598e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vcml_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.894885142184958e-03, -4.931384796694587e-03, 3.042052432208864e-02, 3.039360480947706e-02, 2.104055089883573e-04, 2.262098499906099e-04, -1.080692202285795e-01, 1.529669417213426e-12, 8.666683933259188e-03, 2.397771671342468e-08, -3.384746550564411e-12, 4.925383620046366e-13, 4.399696475523100e-03, 2.671353916447233e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
