
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_scanl_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.221803160260051e+01, -2.221804189427658e+01, -2.221818956095162e+01, -2.221803474750492e+01, -2.221810139372760e+01, -2.221810139372760e+01, -3.703911992633898e+00, -3.703870607206486e+00, -3.702999990098984e+00, -3.789292344661611e+00, -3.703951656462227e+00, -3.788874026494409e+00, -7.278504434521781e-01, -7.299748769035557e-01, -7.105870050129174e-01, -7.164852877981900e-01, -7.152309472104279e-01, -7.152309472104279e-01, -1.847115507472178e-01, -1.915535080046269e-01, -8.484314542186751e-01, -1.432541186914638e-01, -1.608629134505057e-01, -1.608629134505057e-01, -6.568702416665439e-03, -6.913831336724093e-03, -3.814432273319453e-02, -3.793630973909892e-03, -4.766987687188107e-03, -4.766987687188107e-03, -5.672599922386669e+00, -5.673641338130302e+00, -5.672651790450415e+00, -5.277673062949052e+00, -5.673125699780131e+00, -5.673125699780131e+00, -2.118205222806201e+00, -2.133977379166327e+00, -2.106042516443591e+00, -2.119989532300870e+00, -2.132795741627160e+00, -2.146226593591021e+00, -6.392409720547114e-01, -6.944408281973620e-01, -5.914910396723774e-01, -6.148613592662492e-01, -6.495469595366764e-01, -6.231098771992423e-01, -9.153169642404385e-02, -1.889822786652724e-01, -8.405464384170167e-02, -1.799494248942706e+00, -1.167185807330526e-01, -1.167185807330526e-01, -8.804805388219499e-04, -3.708609903802430e-03, -2.835781396072887e-03, -5.400133795568249e-02, -1.194778468358366e-03, -3.416237370976312e-03, -6.423447928852579e-01, -6.386601455997365e-01, -5.577268490121696e-01, -6.410246559926521e-01, -6.404901047791867e-01, -6.404901047791867e-01, -6.249330154065934e-01, -5.364599783685042e-01, -5.617592998206659e-01, -5.868523935651008e-01, -5.740367746088314e-01, -5.530735455658629e-01, -7.280299097353981e-01, -2.333218432895196e-01, -2.867472521537434e-01, -3.769482717382608e-01, -3.293791550482510e-01, -3.293791550482510e-01, -4.919144712988218e-01, -3.650895950044621e-02, -4.977409309957403e-02, -3.651238497945862e-01, -7.164956997527733e-02, -7.880019738340113e-02, -9.270456143909556e-03, -9.913351863349582e-04, -6.668286888150551e-04, -7.399675478779857e-02, -3.161088173721501e-03, -3.161088173721499e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08

# test_mgga_x_scanl_BrOH_cation_restr_1_vrho() not generated due to NaN

# test_mgga_x_scanl_BrOH_cation_restr_1_vsigma() not generated due to NaN

# test_mgga_x_scanl_BrOH_cation_restr_1_vlapl() not generated due to NaN


def test_mgga_x_scanl_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05