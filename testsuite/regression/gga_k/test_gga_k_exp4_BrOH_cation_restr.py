
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_exp4_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_exp4", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.486010506928227e+03, 2.485988251732442e+03, 2.485951538520951e+03, 2.486281397083053e+03, 2.486099696667781e+03, 2.486099696667781e+03, 7.857573868659223e+01, 7.856680298635168e+01, 7.836405781514758e+01, 7.874639894165510e+01, 7.856881298201608e+01, 7.856881298201608e+01, 3.831728636227175e+00, 3.841278213045978e+00, 3.929258980789418e+00, 3.986958508560284e+00, 3.968170854244728e+00, 3.968170854244728e+00, 2.826826052707904e-01, 2.913098752919594e-01, 4.958872133564956e+00, 1.629255598642090e-01, 2.054408711059117e-01, 2.054408711059117e-01, 3.425580378970927e-04, 3.795006718183588e-04, 1.155140482315131e-02, 1.142576145585576e-04, 1.804110142204644e-04, 1.804110142204644e-04, 1.292634122074356e+02, 1.289530125933854e+02, 1.292509513328400e+02, 1.289768668216034e+02, 1.291047687473346e+02, 1.291047687473346e+02, 3.563759964170379e+01, 3.606802566255332e+01, 3.541802532357089e+01, 3.582036766152618e+01, 3.595363345952648e+01, 3.595363345952648e+01, 2.054015052463196e+00, 1.646673360861258e+00, 1.895267184331881e+00, 1.342147706212211e+00, 1.997317261104382e+00, 1.997317261104382e+00, 7.853515016846854e-02, 2.835418378897342e-01, 6.721491069457478e-02, 1.485986091577148e+01, 1.081570874470301e-01, 1.081570874470301e-01, 6.801926487129813e-05, 1.091936278323351e-04, 6.384405833013911e-05, 3.040576227665765e-02, 9.265547057801418e-05, 9.265547057801418e-05, 1.347859794424209e+00, 1.386748228057974e+00, 1.368787028843270e+00, 1.358213492599530e+00, 1.363015708221067e+00, 1.363015708221067e+00, 1.201055562671537e+00, 2.040250727693033e+00, 1.789816894789537e+00, 1.441764773317855e+00, 1.607367321316280e+00, 1.607367321316280e+00, 1.826814822473697e+00, 4.526110271622202e-01, 6.527899541583503e-01, 1.074168633554335e+00, 8.603017962989747e-01, 8.603017962989747e-01, 1.748053380251845e+00, 1.058214965823451e-02, 1.966897368168392e-02, 8.675881958952535e-01, 4.929803380920070e-02, 4.929803380920072e-02, 6.823028652942927e-04, 7.802181161978389e-06, 3.437630673397813e-05, 4.347107049154005e-02, 7.933196427399001e-05, 7.933196427398987e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_exp4_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_exp4", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [8.980365459133825e+02, 8.981093612764184e+02, 8.983258698396511e+02, 8.972449138771008e+02, 8.978264739126093e+02, 8.978264739126093e+02, 9.829204496612560e+00, 9.831515398368616e+00, 9.889031437608994e+00, 9.797211188588577e+00, 9.832450737721883e+00, 9.832450737721883e+00, 2.440000569763224e+00, 2.544792266063094e+00, 5.110281364280465e+00, 5.084854383928659e+00, 5.000305355655552e+00, 5.000305355655552e+00, 4.711376754508939e-01, 4.855164587970185e-01, 2.076043043240546e+00, 2.715425997736816e-01, 3.424014518431863e-01, 3.424014518431863e-01, 5.709300631618211e-04, 6.325011196972646e-04, 1.925234137191886e-02, 1.904293575975960e-04, 3.006850237007740e-04, 3.006850237007740e-04, 9.597824812423613e+01, 9.698963144338542e+01, 9.602033298327102e+01, 9.691326755905200e+01, 9.649345608971433e+01, 9.649345608971433e+01, 5.003788977579527e+01, 4.932075296925736e+01, 5.231414535091339e+01, 5.195442980431910e+01, 4.823363363630708e+01, 4.823363363630708e+01, 4.078695490639670e-01, 2.342113639536403e+00, 2.455994552478689e-01, 1.661747477361440e+00, 5.930303869254001e-01, 5.930303869254001e-01, 1.308919169474476e-01, 4.725697298162237e-01, 1.120248511576246e-01, 2.150908903796878e+01, 1.802618124117168e-01, 1.802618124117168e-01, 1.133654414521635e-04, 1.819893797205585e-04, 1.064067638835652e-04, 5.067627046109609e-02, 1.544257842966903e-04, 1.544257842966903e-04, 2.155101525481871e+00, 2.001955072766357e+00, 2.093363696073160e+00, 2.152758431182976e+00, 2.125306009355657e+00, 2.125306009355657e+00, 1.441945165435760e+00, 1.174126540252119e+00, 2.145876955172309e-01, 8.754603832862724e-01, 3.909674150378636e-01, 3.909674150378636e-01, 2.515305270011077e+00, 7.543517119370305e-01, 1.087976497617994e+00, 9.824376760359524e-01, 1.399951031994842e+00, 1.399951031994842e+00, 1.101945017699745e+00, 1.763691609705753e-02, 3.278162280280653e-02, 2.394959024105285e-01, 8.216338968200118e-02, 8.216338968200119e-02, 1.137171442157155e-03, 1.300363526996398e-05, 5.729384455663022e-05, 7.245178415256677e-02, 1.322199404566500e-04, 1.322199404566498e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_exp4_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_exp4", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [8.216620804354499e-06, 8.216471838495996e-06, 8.215917014192598e-06, 8.218130573699568e-06, 8.216955663259564e-06, 8.216955663259564e-06, 1.938582485271536e-03, 1.938682074385241e-03, 1.940789733265301e-03, 1.935958356708497e-03, 1.938561670073283e-03, 1.938561670073283e-03, 1.407524697916487e-01, 1.366507177074598e-01, 4.314719690631921e-02, 4.552214309878287e-02, 4.787835628307077e-02, 4.787835628307077e-02, 3.138948624677282e-12, 1.688680299624414e-10, 1.149051307998576e-01, 1.163231134948178e-62, 8.180183349809984e-30, 8.180183349809523e-30, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 4.832325143731944e-04, 4.804577125693445e-04, 4.831160764572246e-04, 4.806666454904822e-04, 4.818227192273133e-04, 4.818227192273133e-04, 1.084311115537833e-03, 1.234535989290929e-03, 7.581360876666033e-04, 8.642580973649603e-04, 1.362638332257284e-03, 1.362638332257284e-03, 4.102723745806102e-01, 1.341099247334992e-01, 5.118885201683602e-01, 2.569717762619512e-01, 3.845416970980295e-01, 3.845416970980295e-01, 0.000000000000000e+00, 1.551025490926331e-34, 0.000000000000000e+00, 4.647118261212613e-03, 6.142717047409203e-135, 6.142717047409203e-135, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.318949740775813e-01, 1.646329288985846e-01, 1.281369187472227e-01, 9.891913869755832e-02, 1.130782043524530e-01, 1.130782043524530e-01, 1.961710579989707e+00, 3.838912411998588e-01, 5.815772976509659e-01, 4.666210938930790e-01, 5.634162524034882e-01, 5.634162524034882e-01, 1.268962233132443e-01, 6.327600623735870e-15, 8.906020866520243e-06, 6.782223404742241e-01, 3.511446905616158e-02, 3.511446905616170e-02, 4.596748826147000e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.709195568567221e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05