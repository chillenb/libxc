
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_thakkar_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.182048075540083e+03, 2.182056237075132e+03, 2.182102709826210e+03, 2.181981072369200e+03, 2.182043358800467e+03, 2.182043358800467e+03, 5.916148576316624e+01, 5.916079929734926e+01, 5.914782833999090e+01, 5.919285286022900e+01, 5.916351500291112e+01, 5.916351500291112e+01, 2.309587938101642e+00, 2.307588037959623e+00, 2.266433258422829e+00, 2.300137503830892e+00, 2.289703335522852e+00, 2.289703335522852e+00, 1.928773978910129e-01, 1.959885405859780e-01, 3.102579772963873e+00, 1.328076069948991e-01, 1.535881315152556e-01, 1.535881315152556e-01, 2.281663535834600e-03, 2.398914351247169e-03, 2.231309801719237e-02, 1.217705057463079e-03, 1.515003263990238e-03, 1.515003263990238e-03, 1.274025609639472e+02, 1.273998129298067e+02, 1.274028453176824e+02, 1.274004137049271e+02, 1.274009453045362e+02, 1.274009453045362e+02, 2.057387616905462e+01, 2.080914609542947e+01, 2.050184607338435e+01, 2.070927843771909e+01, 2.073857263480682e+01, 2.073857263480682e+01, 1.665461915596559e+00, 1.855186717594751e+00, 1.439898606486903e+00, 1.467201336590433e+00, 1.706592338139715e+00, 1.706592338139715e+00, 8.218179514115817e-02, 2.153145080654912e-01, 7.310348647775346e-02, 1.680400007006565e+01, 9.724874644461827e-02, 9.724874644461827e-02, 8.500700984565901e-04, 1.146370555819175e-03, 5.721995910867900e-04, 4.087596017708477e-02, 8.338603145559452e-04, 8.338603145559454e-04, 1.562197794389941e+00, 1.567299892747304e+00, 1.565423680789206e+00, 1.563846884252045e+00, 1.564625631774967e+00, 1.564625631774967e+00, 1.473247869600338e+00, 1.240585547285551e+00, 1.306953829107422e+00, 1.371245530192108e+00, 1.337919240971195e+00, 1.337919240971195e+00, 2.043790072414030e+00, 3.140765021534745e-01, 4.149810887735693e-01, 6.293711705235495e-01, 5.079246223290572e-01, 5.079246223290572e-01, 1.054556056316606e+00, 2.144171035276728e-02, 3.199835429150341e-02, 5.642175327302064e-01, 5.489932590521902e-02, 5.489932590521902e-02, 3.602506620576976e-03, 1.790410411609843e-04, 4.888343817589362e-04, 5.051816338976503e-02, 7.253698760860096e-04, 7.253698760860083e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_thakkar_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.276529453960154e+03, 3.276549613944191e+03, 3.276641859038488e+03, 3.276341868432367e+03, 3.276498682611399e+03, 3.276498682611399e+03, 8.630177865716625e+01, 8.630249301722257e+01, 8.632325368935240e+01, 8.631995790761789e+01, 8.630679601433660e+01, 8.630679601433660e+01, 3.181368477476599e+00, 3.173785562595173e+00, 2.997088693283796e+00, 3.047385941359383e+00, 3.036930703678605e+00, 3.036930703678605e+00, 2.153670798022736e-01, 2.212107947618262e-01, 4.333410491291114e+00, 1.321500754990565e-01, 1.609052733335266e-01, 1.609052733335265e-01, 1.350888006696168e-03, 1.431327269886007e-03, 1.666867891457929e-02, 6.778466321598110e-04, 8.686374817329008e-04, 8.686374817329013e-04, 1.958567858003439e+02, 1.959571990127867e+02, 1.958615416016138e+02, 1.959501833712980e+02, 1.959075939871676e+02, 1.959075939871676e+02, 2.697378208066006e+01, 2.737054343912355e+01, 2.668162700474838e+01, 2.702918192026387e+01, 2.733633545655015e+01, 2.733633545655015e+01, 2.463923465409060e+00, 2.972928017734663e+00, 2.104121870485998e+00, 2.320350264534832e+00, 2.549617684659827e+00, 2.549617684659827e+00, 7.301874953958426e-02, 2.233515793428628e-01, 6.404495006559421e-02, 2.698562170294619e+01, 9.206111606909677e-02, 9.206111606909677e-02, 4.647914605163193e-04, 6.392653097358805e-04, 3.253262747409395e-04, 3.348554393250999e-02, 4.738454995860006e-04, 4.738454995860003e-04, 2.571129465698321e+00, 2.516100215985331e+00, 2.534052740026299e+00, 2.549810500308245e+00, 2.541798828462948e+00, 2.541798828462948e+00, 2.445508596651096e+00, 1.715184510527148e+00, 1.895002264223429e+00, 2.091922677938012e+00, 1.989061596789291e+00, 1.989061596789291e+00, 3.263430319076865e+00, 3.463172788704154e-01, 4.905788499437842e-01, 8.531004115530818e-01, 6.433875207082479e-01, 6.433875207082479e-01, 1.453164164491939e+00, 1.584272388756978e-02, 2.491335488323701e-02, 7.969273993651431e-01, 4.769973506446489e-02, 4.769973506446488e-02, 2.210479194191313e-03, 9.248512589072300e-05, 2.636846352845801e-04, 4.326738577545610e-02, 4.113690748783649e-04, 4.113690748783644e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_thakkar_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [9.120131382734589e-07, 9.120108881731607e-07, 9.119898989850129e-07, 9.120236032025249e-07, 9.120075044988338e-07, 9.120075044988338e-07, 1.968616039476913e-04, 1.968699332537265e-04, 1.970482035251085e-04, 1.966264702332196e-04, 1.968573538636373e-04, 1.968573538636373e-04, 2.382406980973206e-02, 2.381327714500970e-02, 2.340503618686059e-02, 2.294018461746769e-02, 2.312558651924206e-02, 2.312558651924206e-02, 7.867808116327242e-01, 7.770392511835426e-01, 1.555072576902094e-02, 1.210101526831656e+00, 1.032504432746724e+00, 1.032504432746724e+00, 3.623061778985747e+02, 3.344039191705843e+02, 1.230401728565202e+01, 9.874926393741209e+02, 6.876762383105162e+02, 6.876762383105162e+02, 6.666678871121207e-05, 6.671545032128906e-05, 6.666848531057567e-05, 6.671145077350139e-05, 6.669174461376168e-05, 6.669174461376168e-05, 8.476960566118055e-04, 8.363254450516483e-04, 8.453094847358469e-04, 8.352604102209855e-04, 8.425879036394743e-04, 8.425879036394743e-04, 4.242826486203882e-02, 3.968207955947532e-02, 5.196045440837539e-02, 5.584079903900699e-02, 4.142270322275293e-02, 4.142270322275293e-02, 2.154339381729182e+00, 6.151351708396989e-01, 2.519814138063164e+00, 1.456682065367195e-03, 1.817304409755250e+00, 1.817304409755250e+00, 1.737867972800986e+03, 1.078559362405438e+03, 2.988556729295353e+03, 5.508783869791373e+00, 1.699769562087694e+03, 1.699769562087692e+03, 4.698280509967882e-02, 5.113508608484236e-02, 5.112354115461903e-02, 5.047221710697369e-02, 5.091337203044063e-02, 5.091337203044063e-02, 3.470759346773766e-02, 6.077621689447796e-02, 5.950431895095682e-02, 5.910705670684593e-02, 5.929810956953144e-02, 5.929810956953144e-02, 3.425052747062993e-02, 3.735411137587923e-01, 2.649103087883151e-01, 1.644265369868811e-01, 2.105228192116003e-01, 2.105228192116003e-01, 7.725102982637760e-02, 1.293610391867757e+01, 7.474213558254704e+00, 2.032009089598689e-01, 3.828900533652008e+00, 3.828900533652006e+00, 1.792158896221682e+02, 2.025378934541792e+04, 4.077600541112016e+03, 4.254320604373128e+00, 2.099651653293341e+03, 2.099651653293344e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05