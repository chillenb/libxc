
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.731384740362278e-02, -2.731386368348848e-02, -2.731348102278690e-02, -2.731322429683826e-02, -2.731342391549214e-02, -2.731342391549214e-02, -4.827268926965755e-02, -4.826888915665621e-02, -4.814673003643273e-02, -4.799256627630424e-02, -4.809444965014567e-02, -4.809444965014567e-02, -4.140771178903926e-02, -4.125202333418644e-02, -3.753237305117307e-02, -3.747585835614704e-02, -3.746619575587877e-02, -3.746619575587877e-02, -1.548166670825183e-02, -1.604897418832450e-02, -4.874036229456221e-02, -1.513444898639855e-02, -1.246015261686435e-02, -1.246015261686435e-02, -6.319290446108716e-04, -6.901861312942174e-04, -4.431961364635226e-03, -3.248915489000223e-04, -3.886736741259402e-04, -3.886736741259402e-04, -2.644926061190398e-02, -2.645704337149266e-02, -2.644845906338973e-02, -2.645537739934189e-02, -2.645380385482305e-02, -2.645380385482305e-02, -2.540011895554291e-02, -2.499577193940631e-02, -2.480807194406921e-02, -2.445164509664020e-02, -2.545368096434297e-02, -2.545368096434297e-02, -2.909535332854004e-02, -3.298535051797552e-02, -3.539394303112370e-02, -4.392829887466025e-02, -2.874334570317248e-02, -2.874334570317245e-02, -1.074663740292649e-02, -1.507186276039064e-02, -1.012927946608019e-02, -5.554886169066260e-02, -1.248174010593854e-02, -1.248174010593854e-02, -2.401899805717472e-04, -3.225259744924791e-04, -2.852294304395399e-04, -7.331303050841491e-03, -3.140580857964027e-04, -3.140580857964000e-04, -2.490774441621052e-02, -2.451317772907648e-02, -2.464870809723436e-02, -2.476360998074802e-02, -2.470633942427986e-02, -2.470633942427986e-02, -2.597059633072191e-02, -2.175247643862722e-02, -2.288063925113423e-02, -2.406556250494013e-02, -2.352528742182709e-02, -2.352528742182709e-02, -3.636876898237665e-02, -1.584094994310955e-02, -1.918784404313855e-02, -2.770143754746717e-02, -2.224712450909592e-02, -2.224712450909592e-02, -2.493050182470805e-02, -4.149887803462375e-03, -5.724625749629297e-03, -2.714954912887881e-02, -9.296397843715717e-03, -9.296397843715720e-03, -9.741404669642745e-04, -6.672422802067299e-05, -1.667619546378284e-04, -8.566974492923641e-03, -2.950488119927136e-04, -2.950488119927060e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.171358916492659e-02, -4.171370143130070e-02, -4.171331216790539e-02, -4.171350147832146e-02, -4.171217179838627e-02, -4.171196250630965e-02, -4.171525178942326e-02, -4.171591586721927e-02, -4.171322596224111e-02, -4.171433710826165e-02, -4.171322596224111e-02, -4.171433710826165e-02, -6.900736770354607e-02, -6.900992293928108e-02, -6.899327533076222e-02, -6.899619325554550e-02, -6.861501759601579e-02, -6.861115955357060e-02, -6.861039086222048e-02, -6.860776991544547e-02, -6.871017076736151e-02, -6.864018595336646e-02, -6.871017076736151e-02, -6.864018595336646e-02, -7.464167212101705e-02, -7.422665182087218e-02, -7.439858824170173e-02, -7.388922278529572e-02, -6.761270575255184e-02, -6.830592871663681e-02, -7.051576647727176e-02, -7.031861098755086e-02, -6.700279089903607e-02, -7.310850346825350e-02, -6.700279089903607e-02, -7.310850346825350e-02, -4.881934017832865e-02, -4.686286505109956e-02, -4.980559196031773e-02, -4.747077930177770e-02, -8.977593513877813e-02, -8.569975906614008e-02, -3.189284399658279e-02, -3.133035366269214e-02, -2.895966024121808e-02, -5.652435155250220e-02, -2.895966024121816e-02, -5.652435155250220e-02, -4.498003077210959e-03, -4.428024270065408e-03, -2.395170597416858e-03, -2.310341146872423e-03, -8.940709555335621e-03, -8.413606336393768e-03, -6.310632280052143e-04, -6.399749540437267e-04, -7.285329723631794e-04, -1.042767599978787e-03, -7.285329723631785e-04, -1.042767599978786e-03, -3.786773572287651e-02, -3.788772282601745e-02, -3.782143287827385e-02, -3.784195310942532e-02, -3.786440681435868e-02, -3.788476477735911e-02, -3.782397461642587e-02, -3.784403142490823e-02, -3.784475644215887e-02, -3.786511350769618e-02, -3.784475644215887e-02, -3.786511350769618e-02, -5.521593159364283e-02, -5.522024896641830e-02, -5.345269688384688e-02, -5.347529055780365e-02, -5.455250736078743e-02, -5.440998776875478e-02, -5.287067315535159e-02, -5.272598837701950e-02, -5.436956922130273e-02, -5.474474857469204e-02, -5.436956922130273e-02, -5.474474857469204e-02, -4.067682161349846e-02, -4.096315576701553e-02, -2.710907346267795e-02, -2.705137818044240e-02, -4.867580877432176e-02, -4.362900102515516e-02, -3.350655531689162e-02, -2.757550808751629e-02, -3.664228219486595e-02, -4.204227557394894e-02, -3.664228219486594e-02, -4.204227557394891e-02, -2.208847265989399e-02, -2.183618978395040e-02, -5.729278301379929e-02, -5.704139036809242e-02, -2.099968069924510e-02, -1.971124489528903e-02, -3.039149636004887e-02, -3.050210738043166e-02, -2.509881465598783e-02, -2.330698344628301e-02, -2.509881465598836e-02, -2.330698344628301e-02, -4.740986813100343e-04, -4.593537739176653e-04, -6.290254342287802e-04, -6.211460703591499e-04, -5.731980085752200e-04, -5.471042736593249e-04, -1.431686912229687e-02, -1.421331584397179e-02, -7.252234303304882e-04, -5.648765776746532e-04, -7.252234303304842e-04, -5.648765776746511e-04, -2.636799887680335e-02, -2.605890488053017e-02, -2.888729458926726e-02, -2.858051595583950e-02, -2.806538919712360e-02, -2.775682958199408e-02, -2.733281400164382e-02, -2.702550366640670e-02, -2.770492481095252e-02, -2.739700951035456e-02, -2.770492481095251e-02, -2.739700951035455e-02, -2.484170817204318e-02, -2.455971788533167e-02, -3.916077124487220e-02, -3.886020117257583e-02, -3.682814457736049e-02, -3.650195770414371e-02, -3.340505298329233e-02, -3.313073793209680e-02, -3.532455288240124e-02, -3.504211787009687e-02, -3.532455288240124e-02, -3.504211787009687e-02, -2.716728870885455e-02, -2.692822947195924e-02, -4.999322468018588e-02, -4.969452190102527e-02, -5.246724869616655e-02, -5.181208117646738e-02, -4.908504676478015e-02, -4.851314122908104e-02, -5.032505113298252e-02, -5.035288069455548e-02, -5.032505113298250e-02, -5.035288069455547e-02, -4.601331754640778e-02, -4.540765615375379e-02, -9.331043791843194e-03, -9.268063783143902e-03, -1.135306169169303e-02, -1.092998164380472e-02, -4.348021783506171e-02, -4.195420987131207e-02, -2.018729598995076e-02, -1.893848506001941e-02, -2.018729598995074e-02, -1.893848506001938e-02, -1.892110921488115e-03, -1.830022954139840e-03, -1.311633679697817e-04, -1.309351743654875e-04, -3.475029102146784e-04, -3.314126770584227e-04, -2.143809606670486e-02, -2.107976168870240e-02, -6.693561879041114e-04, -5.348158686877557e-04, -6.693561879041014e-04, -5.348158686877501e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [8.498401883423958e-11, 1.699680376684791e-10, 8.498401883423958e-11, 8.498389555226323e-11, 1.699677911045265e-10, 8.498389555226323e-11, 8.497737313677428e-11, 1.699547462735486e-10, 8.497737313677428e-11, 8.497911415851185e-11, 1.699582283170237e-10, 8.497911415851185e-11, 8.497905443556131e-11, 1.699581088711227e-10, 8.497905443556131e-11, 8.497905443556131e-11, 1.699581088711227e-10, 8.497905443556131e-11, 1.379429273252066e-06, 2.758858546504131e-06, 1.379429273252066e-06, 1.379736908107304e-06, 2.759473816214609e-06, 1.379736908107304e-06, 1.387012711194424e-06, 2.774025422388848e-06, 1.387012711194424e-06, 1.377404804100937e-06, 2.754809608201874e-06, 1.377404804100937e-06, 1.381684695423616e-06, 2.763369390847232e-06, 1.381684695423616e-06, 1.381684695423616e-06, 2.763369390847232e-06, 1.381684695423616e-06, 1.354661571910559e-03, 2.709323143821118e-03, 1.354661571910559e-03, 1.349084331631808e-03, 2.698168663263616e-03, 1.349084331631808e-03, 1.190466946842203e-03, 2.380933893684406e-03, 1.190466946842203e-03, 1.005957766823330e-03, 2.011915533646660e-03, 1.005957766823330e-03, 1.057515462868157e-03, 2.115030925736313e-03, 1.057515462868157e-03, 1.057515462868157e-03, 2.115030925736313e-03, 1.057515462868157e-03, 5.568892876413202e-01, 1.113778575282640e+00, 5.568892876413202e-01, 5.657925838603817e-01, 1.131585167720763e+00, 5.657925838603817e-01, 8.539274776965107e-04, 1.707854955393021e-03, 8.539274776965107e-04, 3.544119340960271e-01, 7.088238681920542e-01, 3.544119340960271e-01, 5.088182643238740e-01, 1.017636528647748e+00, 5.088182643238740e-01, 5.088182643238751e-01, 1.017636528647750e+00, 5.088182643238751e-01, 1.345846294586545e+03, 2.691692589173089e+03, 1.345846294586545e+03, 4.189395584365637e+02, 8.378791168731273e+02, 4.189395584365637e+02, 2.603273398561425e+00, 5.206546797122851e+00, 2.603273398561425e+00, 1.737212772211823e+02, 3.474425544423646e+02, 1.737212772211823e+02, 1.430180928763450e+02, 2.860361857526898e+02, 1.430180928763450e+02, 1.430180928763442e+02, 2.860361857526883e+02, 1.430180928763442e+02, 1.032060141598068e-07, 2.064120283196137e-07, 1.032060141598068e-07, 1.034797913468253e-07, 2.069595826936506e-07, 1.034797913468253e-07, 1.032049426705671e-07, 2.064098853411341e-07, 1.032049426705671e-07, 1.034470963426540e-07, 2.068941926853079e-07, 1.034470963426540e-07, 1.033516090513945e-07, 2.067032181027890e-07, 1.033516090513945e-07, 1.033516090513945e-07, 2.067032181027890e-07, 1.033516090513945e-07, 8.378772110657499e-06, 1.675754422131500e-05, 8.378772110657499e-06, 7.717246150726427e-06, 1.543449230145285e-05, 7.717246150726427e-06, 7.941069660286958e-06, 1.588213932057392e-05, 7.941069660286958e-06, 7.335910121072382e-06, 1.467182024214476e-05, 7.335910121072382e-06, 8.213575809390722e-06, 1.642715161878144e-05, 8.213575809390722e-06, 8.213575809390722e-06, 1.642715161878144e-05, 8.213575809390722e-06, 6.258349482342059e-03, 1.251669896468412e-02, 6.258349482342059e-03, 6.694373465899639e-03, 1.338874693179927e-02, 6.694373465899639e-03, 1.032285851633209e-02, 2.064571703266417e-02, 1.032285851633209e-02, 1.482040180562904e-02, 2.964080361125809e-02, 1.482040180562904e-02, 5.835084540945574e-03, 1.167016908189115e-02, 5.835084540945574e-03, 5.835084540945570e-03, 1.167016908189114e-02, 5.835084540945570e-03, 5.065032960844955e-01, 1.013006592168991e+00, 5.065032960844955e-01, 4.685272819285828e-01, 9.370545638571657e-01, 4.685272819285828e-01, 5.562042959101746e-01, 1.112408591820349e+00, 5.562042959101746e-01, 6.236411921168749e-05, 1.247282384233750e-04, 6.236411921168749e-05, 3.139903629629793e-01, 6.279807259259585e-01, 3.139903629629793e-01, 3.139903629629603e-01, 6.279807259259206e-01, 3.139903629629603e-01, 2.951388792121642e+02, 5.902777584243285e+02, 2.951388792121642e+02, 1.865646950520859e+02, 3.731293901041719e+02, 1.865646950520859e+02, 9.764573857324565e+02, 1.952914771464913e+03, 9.764573857324565e+02, 1.185192316443767e+00, 2.370384632887535e+00, 1.185192316443767e+00, 4.004211037195153e+02, 8.008422074390305e+02, 4.004211037195153e+02, 4.004211037195151e+02, 8.008422074390303e+02, 4.004211037195151e+02, 5.737471676170253e-03, 1.147494335234051e-02, 5.737471676170253e-03, 5.248904903614629e-03, 1.049780980722926e-02, 5.248904903614629e-03, 5.410241076472548e-03, 1.082048215294510e-02, 5.410241076472548e-03, 5.551830342702013e-03, 1.110366068540403e-02, 5.551830342702013e-03, 5.480372862546982e-03, 1.096074572509396e-02, 5.480372862546982e-03, 5.480372862546982e-03, 1.096074572509396e-02, 5.480372862546982e-03, 7.483769355776096e-03, 1.496753871155219e-02, 7.483769355776096e-03, 6.256155698055975e-03, 1.251231139611195e-02, 6.256155698055975e-03, 6.295427727114861e-03, 1.259085545422972e-02, 6.295427727114861e-03, 6.574129774617402e-03, 1.314825954923481e-02, 6.574129774617402e-03, 6.466589128430577e-03, 1.293317825686115e-02, 6.466589128430577e-03, 6.466589128430578e-03, 1.293317825686115e-02, 6.466589128430578e-03, 6.028112786598823e-03, 1.205622557319765e-02, 6.028112786598823e-03, 1.644216060018419e-01, 3.288432120036838e-01, 1.644216060018419e-01, 1.054261245930304e-01, 2.108522491860608e-01, 1.054261245930304e-01, 5.555996511445374e-02, 1.111199302289075e-01, 5.555996511445374e-02, 7.533799589101448e-02, 1.506759917820290e-01, 7.533799589101448e-02, 7.533799589101443e-02, 1.506759917820289e-01, 7.533799589101443e-02, 1.441181437647194e-02, 2.882362875294387e-02, 1.441181437647194e-02, 4.813545825954206e+00, 9.627091651908412e+00, 4.813545825954206e+00, 1.485156516934691e+00, 2.970313033869382e+00, 1.485156516934691e+00, 7.671336481838699e-02, 1.534267296367740e-01, 7.671336481838699e-02, 1.252579314803593e+00, 2.505158629607186e+00, 1.252579314803593e+00, 1.252579314803584e+00, 2.505158629607169e+00, 1.252579314803584e+00, 3.164820528367225e+01, 6.329641056734451e+01, 3.164820528367225e+01, 4.385142195829963e+03, 8.770284391659925e+03, 4.385142195829963e+03, 1.096905545768600e+03, 2.193811091537201e+03, 1.096905545768600e+03, 2.363050598713810e+00, 4.726101197427621e+00, 2.363050598713810e+00, 5.449565062673246e+02, 1.089913012534649e+03, 5.449565062673246e+02, 5.449565062673285e+02, 1.089913012534657e+03, 5.449565062673285e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.128695423902140e-05, -1.128695423902140e-05, -1.128705229454227e-05, -1.128705229454227e-05, -1.128686229083060e-05, -1.128686229083059e-05, -1.128539714756280e-05, -1.128539714756279e-05, -1.128625407474892e-05, -1.128625407474892e-05, -1.128625407474892e-05, -1.128625407474892e-05, -4.977718081762686e-04, -4.977718081762684e-04, -4.979951587453104e-04, -4.979951587453096e-04, -5.039194957886528e-04, -5.039194957886530e-04, -5.022920689890062e-04, -5.022920689890069e-04, -5.022152880428358e-04, -5.022152880428355e-04, -5.022152880428358e-04, -5.022152880428355e-04, -6.802200461898598e-04, -6.802200461899113e-04, -6.984385944222445e-04, -6.984385944223227e-04, -9.564220421013108e-04, -9.564220421013865e-04, -3.029490190697936e-04, -3.029490190697936e-04, -4.627708452472143e-04, -4.627708452471114e-04, -4.627708452472143e-04, -4.627708452471114e-04, -4.037109144803300e-02, -4.037109144803300e-02, -4.276098925399066e-02, -4.276098925399069e-02, -3.881471635792507e-04, -3.881471635792507e-04, -6.925501313301424e-03, -6.925501313301426e-03, -2.068730928093244e-02, -2.068730928093235e-02, -2.068730928093241e-02, -2.068730928093246e-02, -4.432442290030502e-03, -4.432442290030507e-03, -1.436037407453977e-03, -1.436037407453978e-03, -4.808478229704727e-04, -4.808478229704727e-04, -8.884142498726826e-06, -8.884142498726826e-06, -4.353772972558583e-05, -4.353772972558588e-05, -4.353772972558461e-05, -4.353772972558463e-05, -2.059712540725588e-04, -2.059712540725588e-04, -2.066934544541581e-04, -2.066934544541581e-04, -2.059876719046061e-04, -2.059876719046061e-04, -2.066257441286905e-04, -2.066257441286906e-04, -2.063453774379431e-04, -2.063453774379430e-04, -2.063453774379431e-04, -2.063453774379430e-04, -8.267434686584205e-04, -8.267434686584209e-04, -7.799330987460291e-04, -7.799330987460286e-04, -7.741456396491430e-04, -7.741456396491430e-04, -7.300808668913415e-04, -7.300808668913416e-04, -8.260029618572196e-04, -8.260029618572196e-04, -8.260029618572196e-04, -8.260029618572196e-04, -1.551049255316465e-02, -1.551049255316465e-02, -1.970365695762902e-02, -1.970365695762904e-02, -1.784657610276065e-02, -1.784657610276066e-02, -2.394503560165308e-02, -2.394503560165310e-02, -1.520908799890077e-02, -1.520908799890077e-02, -1.520908799890076e-02, -1.520908799890076e-02, -2.736460487856742e-03, -2.736460487856744e-03, -3.479652459931598e-02, -3.479652459931602e-02, -2.075362390679529e-03, -2.075362390679530e-03, -3.428147683371157e-03, -3.428147683371157e-03, -1.275892182173858e-03, -1.275892182174501e-03, -1.275892182174807e-03, -1.275892182174455e-03, -3.747646031973689e-07, -3.747646031973688e-07, -1.639051109737151e-06, -1.639051109737151e-06, -1.702670618336913e-05, -1.702670618336912e-05, -9.365271717082538e-04, -9.365271717082541e-04, -4.026501816094015e-06, -4.026501816094013e-06, -4.026501816093969e-06, -4.026501816093969e-06, -1.603268234278722e-02, -1.603268234278723e-02, -1.436646311912300e-02, -1.436646311912300e-02, -1.491938802993376e-02, -1.491938802993377e-02, -1.540312314425545e-02, -1.540312314425546e-02, -1.515876435792477e-02, -1.515876435792476e-02, -1.515876435792477e-02, -1.515876435792476e-02, -1.861533605675382e-02, -1.861533605675382e-02, -9.931350443185416e-03, -9.931350443185417e-03, -1.148139135884173e-02, -1.148139135884174e-02, -1.362459433129808e-02, -1.362459433129808e-02, -1.255982475876872e-02, -1.255982475876872e-02, -1.255982475876872e-02, -1.255982475876872e-02, -1.932538978470272e-02, -1.932538978470273e-02, -2.415404951051975e-02, -2.415404951051975e-02, -2.616715990735462e-02, -2.616715990735462e-02, -2.815073696547810e-02, -2.815073696547812e-02, -2.738322499627229e-02, -2.738322499627228e-02, -2.738322499627228e-02, -2.738322499627226e-02, -1.732247160901572e-02, -1.732247160901572e-02, -1.720081330706958e-03, -1.720081330706960e-03, -5.934749349565714e-04, -5.934749349565712e-04, -3.542148270561645e-02, -3.542148270561645e-02, -3.825706597497304e-03, -3.825706597497306e-03, -3.825706597497256e-03, -3.825706597497258e-03, -4.393352393481641e-06, -4.393352393481641e-06, -2.739231759403271e-08, -2.739231759403271e-08, -2.834999508921695e-05, -2.834999508921694e-05, -8.424939962820607e-03, -8.424939962820606e-03, -8.316266965718108e-06, -8.316266965718108e-06, -8.316266965717649e-06, -8.316266965717654e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05