
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rmsb86bl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.847031493492079e+00, -1.235323985383276e+00, -2.672416042321614e-01, -1.685958840076986e-01, -5.839230992630868e-02, -6.015107247412621e-02, -1.964841620305952e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rmsb86bl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.556216698449962e+00, -2.558580039102003e+00, -1.792922971271082e+00, -1.794765056657864e+00, -3.606451409575730e-01, -3.607889893795037e-01, -2.282802246034018e-01, -3.808704741162946e-02, -8.167212671686998e-02, -2.949692236443671e-03, -6.846931269161007e-02, -3.945567857480805e-02, -2.144588022193127e-02, -2.006764090034732e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsb86bl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.977761879746480e-04, 0.000000000000000e+00, -3.961815242113338e-04, -1.876604958295542e-03, 0.000000000000000e+00, -1.873267053401042e-03, -2.343031094714345e-01, 0.000000000000000e+00, -2.348627815189672e-01, -4.550084117830401e+00, 0.000000000000000e+00, -1.531959616140053e+02, -1.189247787046829e+02, 0.000000000000000e+00, -1.262392476635947e+06, -6.043944725126529e+00, 0.000000000000000e+00, -1.353797428808281e+02, -2.429764187646105e+02, 0.000000000000000e+00, -2.269977851092748e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsb86bl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.350175096861292e-02, 1.348115262211374e-02, 2.144221351190459e-02, 2.148086194725527e-02, 7.793571466759304e-04, 8.383362067801131e-04, 9.755022235811134e-02, 1.331487878863168e-16, 3.194874561787618e-02, 2.324389430792247e-19, -1.502694868879416e-20, 4.877779946328005e-17, 1.487620431708694e-29, 9.672094433121441e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
