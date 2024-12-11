
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_b94_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.490310332403873e-02, -4.625153314305778e-02, -7.435489068897475e-03, -7.957127468107150e-05, -3.793249656734052e-09, -1.059415131174904e-03, -3.794362067238579e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_b94_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.086597628280329e-01, -1.084583513721225e-01, -4.403912463763657e-02, -4.401362019197232e-02, -2.416561631391298e-02, -2.389227187649082e-02, -1.248941899498990e-03, -5.933462446985951e-02, -2.940502972831827e-03, -4.178243148157026e-02, -7.164969022514718e-03, -6.298425357089565e-05, -2.517318841781176e-02, -2.726961579828643e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.430030635091991e-05, 0.000000000000000e+00, 5.405890825211776e-05, 2.350293018329490e-04, 0.000000000000000e+00, 2.335329136822762e-04, 1.203667894806330e-02, 0.000000000000000e+00, 1.185677959378024e-02, 1.733151280956608e+00, 0.000000000000000e+00, 2.309369543670153e+02, 1.715866117074662e+01, 0.000000000000000e+00, 2.556163656445206e+07, 4.690812564569858e+01, 0.000000000000000e+00, 1.013718743687688e+00, 6.683817683302496e+07, 0.000000000000000e+00, 4.682996222901833e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([1.916017025680366e-06, 1.912231848704955e-06, 5.677158580750868e-04, 5.648368491340944e-04, 2.471899566683088e-04, 2.367650056872445e-04, 5.229083439298317e-05, 7.342727697471237e-04, 2.745686181240986e-09, 2.603657800727735e-03, 1.705732685749077e-04, 2.593630705339738e-07, 2.028799729524078e-03, 2.891791128346501e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b94_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.817584957260093e-03, -2.812648202930038e-03, -4.081544828340416e-03, -4.065922780037664e-03, -2.898025576505067e-03, -2.851071354528705e-03, -6.653088912778737e-02, -2.949457368802164e-03, -4.103452852214548e-02, -1.041471598910032e-02, -6.966406996258053e-04, -1.472880922110926e-05, -8.115227710324103e-03, -2.042872647580089e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
