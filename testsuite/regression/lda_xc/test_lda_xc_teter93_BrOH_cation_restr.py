
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_teter93_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.996598763016869e+01, -1.996604293654147e+01, -1.996630588483698e+01, -1.996548268244733e+01, -1.996591159697595e+01, -1.996591159697595e+01, -3.339451133435792e+00, -3.339458155667520e+00, -3.339700185707476e+00, -3.339897192184228e+00, -3.339537980249306e+00, -3.339537980249306e+00, -6.892055877856331e-01, -6.884916915614062e-01, -6.713060043016028e-01, -6.765357783688258e-01, -6.754237116709544e-01, -6.754237116709544e-01, -1.985729576693982e-01, -2.012814196904529e-01, -7.964826524880009e-01, -1.550050477602567e-01, -1.720110160123295e-01, -1.720110160123295e-01, -9.380167412845052e-03, -9.842291741727004e-03, -4.724056553472358e-02, -5.574361487009851e-03, -6.929282136206878e-03, -6.929282136206878e-03, -4.953957819715613e+00, -4.954849032829431e+00, -4.954002206912612e+00, -4.954788932601810e+00, -4.954407764758588e+00, -4.954407764758588e+00, -1.912440932798293e+00, -1.925822590304852e+00, -1.902670778081133e+00, -1.914497625887482e+00, -1.924561337168885e+00, -1.924561337168885e+00, -6.079381240698278e-01, -6.569831519113679e-01, -5.654035158989683e-01, -5.862348210391356e-01, -6.171047078029579e-01, -6.171047078029579e-01, -1.117731638995004e-01, -1.988446881426747e-01, -1.042471959183225e-01, -1.877155311228528e+00, -1.289988986501522e-01, -1.289988986501522e-01, -4.346372648170685e-03, -5.454836846120092e-03, -4.215744649794692e-03, -7.304725535353673e-02, -5.042271127352315e-03, -5.042271127352315e-03, -6.106992853738289e-01, -6.074213729658955e-01, -6.085748887945210e-01, -6.095249427283429e-01, -6.090494047582088e-01, -6.090494047582088e-01, -5.952041621829773e-01, -5.162379046470581e-01, -5.388615349321606e-01, -5.612655840483014e-01, -5.498275293642150e-01, -5.498275293642150e-01, -6.867682583877878e-01, -2.456231908968030e-01, -2.900746817047754e-01, -3.741778478827557e-01, -3.291982604726176e-01, -3.291982604726176e-01, -4.780784546764634e-01, -4.540613035040744e-02, -6.005544440964776e-02, -3.617922139621349e-01, -9.072935762671659e-02, -9.072935762671658e-02, -1.294497403415822e-02, -1.512527179761583e-03, -3.124576504807582e-03, -8.575508291693100e-02, -4.680213547385207e-03, -4.680213547385202e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_teter93_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_teter93", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.657635373615265e+01, -2.657642746297176e+01, -2.657677798933063e+01, -2.657568060958057e+01, -2.657625237919940e+01, -2.657625237919940e+01, -4.423931560514106e+00, -4.423940899269794e+00, -4.424262771248866e+00, -4.424524767136030e+00, -4.424047056730886e+00, -4.424047056730886e+00, -9.050381117512351e-01, -9.040932246288635e-01, -8.813483425007975e-01, -8.882695313588244e-01, -8.867977772582183e-01, -8.867977772582183e-01, -2.582202973578806e-01, -2.617606113691896e-01, -1.047074660945818e+00, -2.013674777464197e-01, -2.235367454525096e-01, -2.235367454525096e-01, -1.231891997760337e-02, -1.292093681268281e-02, -6.146882103379550e-02, -7.350614170966723e-03, -9.122015253062148e-03, -9.122015253062148e-03, -6.572107151118232e+00, -6.573293433107552e+00, -6.572166234318886e+00, -6.573213434475794e+00, -6.572706066960220e+00, -6.572706066960220e+00, -2.527428924102695e+00, -2.545198886281466e+00, -2.514455011803025e+00, -2.530160055704443e+00, -2.543524010730169e+00, -2.543524010730169e+00, -7.975091967977247e-01, -8.623945969782562e-01, -7.412612191251388e-01, -7.688055579990447e-01, -8.096342374554696e-01, -8.096342374554696e-01, -1.451426123522178e-01, -2.585754546969584e-01, -1.353732484454416e-01, -2.480573548425879e+00, -1.675228566129285e-01, -1.675228566129285e-01, -5.741582648106597e-03, -7.194166963022966e-03, -5.570177952975764e-03, -9.491546651805302e-02, -6.653894280458366e-03, -6.653894280458366e-03, -8.011613963027253e-01, -7.968256985529037e-01, -7.983514397000664e-01, -7.996080768053221e-01, -7.989790809404794e-01, -7.989790809404794e-01, -7.806671850256864e-01, -6.762782693322711e-01, -7.061753132008368e-01, -7.357905594850616e-01, -7.206699425307368e-01, -7.206699425307368e-01, -9.018121707725867e-01, -3.198090909735379e-01, -3.781447269919782e-01, -4.888079394236471e-01, -4.295831688682404e-01, -4.295831688682405e-01, -6.258724596376741e-01, -5.909050590567636e-02, -7.807887802097521e-02, -4.724915897484204e-01, -1.178372689542550e-01, -1.178372689542549e-01, -1.695872282357037e-02, -2.008917743549947e-03, -4.136210290012248e-03, -1.113874938887432e-01, -6.179408737194195e-03, -6.179408737194191e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05