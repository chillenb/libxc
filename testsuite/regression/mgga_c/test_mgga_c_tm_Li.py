
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.447834352001868e-02, -2.764059596275451e-02, -2.654662973748058e-03, -2.453417410541883e-03, -5.294125595584699e-09, -7.607968796421618e-09, -1.790424362528482e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.001434984380983e-01, -1.000420545290081e-01, -8.246217509816486e-02, -8.237782702234832e-02, -1.427130361684859e-02, -1.429109085619988e-02, -2.912144133457650e-02, -1.495000708005108e-01, -2.696300044164072e-03, -4.611018807722143e-03, -4.924111261378675e-08, -4.948962448436904e-08, -1.136104462411605e-15, -1.343836472708371e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.000100785776410e-04, 2.818302390492451e-04, 1.001175390821415e-04, 2.111413059572539e-04, 6.645175064483746e-04, 2.114723363155931e-04, 2.705006550319339e-03, 7.265973410204581e-03, 2.712497012171260e-03, 3.294963787361971e+01, 7.322737669062350e+01, 7.815248073389709e+01, 1.573356208637036e+01, 3.146497316282185e+01, 6.328498470999668e+03, 1.678222871991258e-04, 3.357815732941035e-04, 1.678311487030971e-04, 1.606947891974255e-06, 3.213895784120031e-06, 1.606947891982351e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.089846531422426e-02, -1.089846531422426e-02, -7.241125689473858e-03, -7.241125689473857e-03, -2.623033533272320e-04, -2.623033533272318e-04, -1.138569462618429e+00, -1.138569462618177e+00, -3.762626910431491e-02, -3.762626907391640e-02, -4.525376246325964e-14, -4.525376246325965e-14, -9.070442358332183e-32, -9.070442358332188e-32])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
