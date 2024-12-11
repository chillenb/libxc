
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_msb86bl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.847031482842830e+00, -1.235323985383276e+00, -2.672416042321614e-01, -1.685958840076986e-01, -5.839230992630868e-02, -6.015107247412621e-02, -1.964841620305952e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_msb86bl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.556216788643531e+00, -2.558580039102003e+00, -1.792922971271082e+00, -1.794765056657864e+00, -3.606451409575730e-01, -3.607889893795037e-01, -2.282802246034018e-01, -3.808704741161564e-02, -8.167212671686998e-02, -2.949692235391973e-03, -6.846931269171438e-02, -3.945567857480337e-02, -2.144478473521755e-02, -2.006764090034732e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msb86bl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.977071955402450e-04, 0.000000000000000e+00, -3.961127135977355e-04, -1.875941774924398e-03, 0.000000000000000e+00, -1.872606018908871e-03, -2.342788662111237e-01, 0.000000000000000e+00, -2.348372983978579e-01, -4.549670785431033e+00, 0.000000000000000e+00, -1.531959616143591e+02, -1.188777979224473e+02, 0.000000000000000e+00, -1.262392478768817e+06, -6.043944725079153e+00, 0.000000000000000e+00, -1.353797428809309e+02, -2.430267865661309e+02, 0.000000000000000e+00, -2.269977851092748e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msb86bl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.350175977045206e-02, 1.348115262211374e-02, 2.144221351190459e-02, 2.148086194725527e-02, 7.793571466759304e-04, 8.383362067801131e-04, 9.755022235811134e-02, 4.651148124961707e-15, 3.194874561787618e-02, 8.690071040431456e-13, -7.035939259316857e-16, 1.543396654983670e-15, 6.115459726377587e-12, 9.489552485561723e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
