
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tm_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.226541052032837e+01, -2.226546410400093e+01, -2.226576819081281e+01, -2.226497146603053e+01, -2.226537951005901e+01, -2.226537951005901e+01, -3.402817740785241e+00, -3.402822952896122e+00, -3.403201962119417e+00, -3.405075394277314e+00, -3.403805340416927e+00, -3.403805340416927e+00, -6.699638138218613e-01, -6.697267956813491e-01, -6.646880384144365e-01, -6.684365876288989e-01, -6.697253371656156e-01, -6.697253371656156e-01, -2.014095548174283e-01, -2.029712313750684e-01, -8.279725962865305e-01, -1.608119133390941e-01, -1.945057893837962e-01, -1.945057893837961e-01, -2.023003947175402e-02, -2.072874724100553e-02, -6.305653129060276e-02, -1.426003430292765e-02, -1.728800643463440e-02, -1.728800643463438e-02, -5.413798531412823e+00, -5.414335667399268e+00, -5.413835752141227e+00, -5.414309438015492e+00, -5.414063922172769e+00, -5.414063922172769e+00, -2.105103393336512e+00, -2.121805662257845e+00, -2.103542148929530e+00, -2.118366016723914e+00, -2.115125727705969e+00, -2.115125727705969e+00, -5.907493167824598e-01, -6.103202002005375e-01, -5.362382364412546e-01, -5.343862525731298e-01, -6.001472185429070e-01, -6.001472185429071e-01, -1.240050283160564e-01, -2.101121869672274e-01, -1.166041107107795e-01, -1.808611486797364e+00, -1.380129969315321e-01, -1.380129969315321e-01, -1.227522205666320e-02, -1.377288628128263e-02, -9.802802283103691e-03, -8.598605819718461e-02, -1.264727019095337e-02, -1.264727019095338e-02, -5.887478317557699e-01, -6.014216396538226e-01, -5.999271762544673e-01, -5.970445714347047e-01, -5.987462078844107e-01, -5.987462078844107e-01, -5.468517570185734e-01, -5.241745412786062e-01, -5.406057362915795e-01, -5.566532174124372e-01, -5.482421847869301e-01, -5.482421847869301e-01, -6.368276797077501e-01, -2.569187874087775e-01, -2.955063122754739e-01, -3.590971164606580e-01, -3.267615344501024e-01, -3.267615344501025e-01, -4.740438369827140e-01, -6.234388381548269e-02, -7.562546634516232e-02, -3.415570347647061e-01, -1.013333831777980e-01, -1.013333831777980e-01, -2.479326107587771e-02, -6.936222337787554e-03, -9.054803054198396e-03, -9.762085153357489e-02, -1.162273337397352e-02, -1.162273337397351e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tm_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.792926936041279e+01, -2.793103621528953e+01, -2.792941875194936e+01, -2.793114452994277e+01, -2.793009234340449e+01, -2.793205496868849e+01, -2.792841782760473e+01, -2.792991672492852e+01, -2.792935547933076e+01, -2.793091396284236e+01, -2.792935547933076e+01, -2.793091396284236e+01, -4.153054259407992e+00, -4.153253825232539e+00, -4.153103658961161e+00, -4.153310396122200e+00, -4.154368491624133e+00, -4.154718592515710e+00, -4.152940756912290e+00, -4.153287304926081e+00, -4.152248741278884e+00, -4.154480079215395e+00, -4.152248741278884e+00, -4.154480079215395e+00, -7.754976736333570e-01, -7.788268195544167e-01, -7.742720182278918e-01, -7.782529242290381e-01, -7.588857675367547e-01, -7.542247402518748e-01, -7.614814053383772e-01, -7.630345386674969e-01, -7.844705587414192e-01, -7.342351719273639e-01, -7.844705587414192e-01, -7.342351719273639e-01, -1.940327405898114e-01, -2.032768054111595e-01, -1.964264781594681e-01, -2.070595817050176e-01, -8.813823574789930e-01, -8.979729393308339e-01, -1.592744058042414e-01, -1.540931353744703e-01, -2.025265253161209e-01, -1.375607085165254e-01, -2.025265253161209e-01, -1.375607085165253e-01, -1.052981777914724e-02, -1.084255768699252e-02, -1.096214742828319e-02, -1.130578984386233e-02, -4.640359582591814e-02, -4.719217528087292e-02, -9.603442799065734e-03, -9.147881902567516e-03, -1.022029130003721e-02, -9.890750669328575e-03, -1.022029130003722e-02, -9.890750669328572e-03, -7.015411111408975e+00, -7.014343537473739e+00, -7.018975220149321e+00, -7.017791867394857e+00, -7.015886180768677e+00, -7.014661604197213e+00, -7.018935435581790e+00, -7.017785107072244e+00, -7.017026528858438e+00, -7.016040650268506e+00, -7.017026528858438e+00, -7.016040650268506e+00, -2.325407354129271e+00, -2.327500852229154e+00, -2.355936226571176e+00, -2.357133646500874e+00, -2.314162037678470e+00, -2.319236371535017e+00, -2.341124531497070e+00, -2.346733471880357e+00, -2.351201526344950e+00, -2.344303570068595e+00, -2.351201526344950e+00, -2.344303570068595e+00, -7.088675627164170e-01, -7.077261712456063e-01, -7.933423928891716e-01, -7.937895235013581e-01, -6.355539964789978e-01, -6.617213283951643e-01, -6.862726176934085e-01, -7.107436574117829e-01, -7.441917318097632e-01, -7.019767422463153e-01, -7.441917318097634e-01, -7.019767422463155e-01, -1.109819812097995e-01, -1.130885729109559e-01, -1.969584032292178e-01, -1.979927592605143e-01, -1.042855977255516e-01, -1.063832654903966e-01, -2.389128480571726e+00, -2.388098988916725e+00, -1.218670762854673e-01, -1.267818671902254e-01, -1.218670762854672e-01, -1.267818671902252e-01, -1.398044339557230e-02, -1.396804853106005e-02, -1.187737225613262e-02, -1.263050331186946e-02, -6.577425430102002e-03, -6.847185471459975e-03, -7.715151035060470e-02, -7.118668835579281e-02, -1.469086620708911e-02, -7.073202186287661e-03, -1.469086620708911e-02, -7.073202186287646e-03, -7.343339086452646e-01, -7.371601015880980e-01, -7.587285811265785e-01, -7.621619162152615e-01, -7.527236902211257e-01, -7.559891103143077e-01, -7.450987194163978e-01, -7.482567903713897e-01, -7.491851974534512e-01, -7.524086310757201e-01, -7.491851974534512e-01, -7.524086310757201e-01, -7.171558968639714e-01, -7.194031277749356e-01, -6.128124776823132e-01, -6.160901917805695e-01, -6.532416161195805e-01, -6.569136567448723e-01, -6.895902786712242e-01, -6.922399262414765e-01, -6.711256963143867e-01, -6.745590834969357e-01, -6.711256963143867e-01, -6.745590834969356e-01, -8.309551189327543e-01, -8.326277062646986e-01, -2.503252586628390e-01, -2.518227703168857e-01, -3.013717665434207e-01, -3.047027481082079e-01, -4.026327337576590e-01, -4.048934052314994e-01, -3.502321737967470e-01, -3.501678975001949e-01, -3.502321737967470e-01, -3.501678975001949e-01, -5.357476703190545e-01, -5.420608643898792e-01, -4.154894486746016e-02, -4.237704481821584e-02, -5.962906829219517e-02, -6.076240485183088e-02, -3.910686397445172e-01, -4.012567875670707e-01, -8.607959104618859e-02, -8.939224875567336e-02, -8.607959104618861e-02, -8.939224875567335e-02, -2.211023196192554e-02, -2.329966954296679e-02, -8.503459353574759e-03, -8.630875819444503e-03, -5.141937218263928e-03, -5.336529683387509e-03, -8.110529988877883e-02, -7.868966754945468e-02, -1.201017640239258e-02, -6.674786404536243e-03, -1.201017640239258e-02, -6.674786404536242e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.639787990870277e-08, 0.000000000000000e+00, -1.637414062485625e-08, -1.639758177348990e-08, 0.000000000000000e+00, -1.637392608182364e-08, -1.639329824226795e-08, 0.000000000000000e+00, -1.636871776192095e-08, -1.639651775166039e-08, 0.000000000000000e+00, -1.637287482521233e-08, -1.639777886589357e-08, 0.000000000000000e+00, -1.636878300557468e-08, -1.639777886589357e-08, 0.000000000000000e+00, -1.636878300557468e-08, -2.312232845483016e-05, 0.000000000000000e+00, -2.308466548377119e-05, -2.313206722301065e-05, 0.000000000000000e+00, -2.308807927367887e-05, -2.328738726591694e-05, 0.000000000000000e+00, -2.330566314892521e-05, -2.336385164646331e-05, 0.000000000000000e+00, -2.333221929483318e-05, -2.311866139869780e-05, 0.000000000000000e+00, -2.347347715922026e-05, -2.311866139869780e-05, 0.000000000000000e+00, -2.347347715922026e-05, -5.009516352167865e-03, 0.000000000000000e+00, -5.022904617795134e-03, -5.033503410724993e-03, 0.000000000000000e+00, -5.010725122684279e-03, -5.131814215018887e-03, 0.000000000000000e+00, -5.588338623008154e-03, -6.390261185471199e-03, 0.000000000000000e+00, -6.513778198143664e-03, -4.619921693254082e-03, 0.000000000000000e+00, -8.460838346412371e-03, -4.619921693254082e-03, 0.000000000000000e+00, -8.460838346412371e-03, -1.733781578403255e+00, 0.000000000000000e+00, -1.586360940743186e+00, -1.710865725674008e+00, 0.000000000000000e+00, -1.548958929815760e+00, 5.142689765280457e-04, 0.000000000000000e+00, 2.809061994104888e-05, -1.889831581522780e+00, 0.000000000000000e+00, -2.538871144986972e+00, -1.476355305201472e+00, 0.000000000000000e+00, -4.776044382468786e-01, -1.476355305201471e+00, 0.000000000000000e+00, -4.776044382468598e-01, -8.218827003082584e+03, 0.000000000000000e+00, -7.190855625755089e+03, -7.343335066786101e+03, 0.000000000000000e+00, -6.347863982932539e+03, -6.613876295058539e+01, 0.000000000000000e+00, -6.248555268893301e+01, -1.824138654412518e+04, 0.000000000000000e+00, -2.045872232471452e+04, -1.012110931803039e+04, 0.000000000000000e+00, -1.887717796471317e+04, -1.012110931803037e+04, 0.000000000000000e+00, -1.887717796471294e+04, -2.336095135064611e-06, 0.000000000000000e+00, -2.322720289186881e-06, -2.315337181353918e-06, 0.000000000000000e+00, -2.302424689099940e-06, -2.328516348478247e-06, 0.000000000000000e+00, -2.317388150468299e-06, -2.310878761646578e-06, 0.000000000000000e+00, -2.299138467647211e-06, -2.330399704177984e-06, 0.000000000000000e+00, -2.313378168522383e-06, -2.330399704177984e-06, 0.000000000000000e+00, -2.313378168522383e-06, -1.936368580954223e-04, 0.000000000000000e+00, -1.927113449402943e-04, -1.859200076697571e-04, 0.000000000000000e+00, -1.851832555458150e-04, -1.871579896046014e-04, 0.000000000000000e+00, -1.882237255938842e-04, -1.801112454520636e-04, 0.000000000000000e+00, -1.809051427450302e-04, -1.941747199062108e-04, 0.000000000000000e+00, -1.894556657406955e-04, -1.941747199062108e-04, 0.000000000000000e+00, -1.894556657406955e-04, -5.017665627930652e-02, 0.000000000000000e+00, -5.131582577767728e-02, -4.302447990413451e-02, 0.000000000000000e+00, -4.513373058166956e-02, -4.600117383484748e-02, 0.000000000000000e+00, -5.059644872037741e-02, -2.916331904257956e-02, 0.000000000000000e+00, -3.272030068268988e-02, -4.842824299431504e-02, 0.000000000000000e+00, -5.417004049800231e-02, -4.842824299431503e-02, 0.000000000000000e+00, -5.417004049800223e-02, -5.264421409458921e+00, 0.000000000000000e+00, -4.929727413639074e+00, -1.218832214752476e+00, 0.000000000000000e+00, -1.198886223111856e+00, -5.989858657193656e+00, 0.000000000000000e+00, -6.009371616499557e+00, -2.340277173073352e-04, 0.000000000000000e+00, -2.345173278868386e-04, -4.697564592709617e+00, 0.000000000000000e+00, -4.874942955697358e+00, -4.697564592709612e+00, 0.000000000000000e+00, -4.874942955697363e+00, 1.980061699709397e+04, 0.000000000000000e+00, 1.329716617880713e+04, -8.663751521277920e+03, 0.000000000000000e+00, -5.814918342477081e+03, -8.529469815686128e+04, 0.000000000000000e+00, -7.501746164319856e+04, -1.429193052245617e+01, 0.000000000000000e+00, -2.003745657799055e+01, 5.307839117946291e+04, 0.000000000000000e+00, -4.130900620611753e+04, 5.307839117946265e+04, 0.000000000000000e+00, -4.130900620611758e+04, -3.911723204076080e-01, 0.000000000000000e+00, -3.965262314033398e-01, -1.184537605288993e-01, 0.000000000000000e+00, -1.179611732425137e-01, -1.695968545240706e-01, 0.000000000000000e+00, -1.700906583257046e-01, -2.426472865636874e-01, 0.000000000000000e+00, -2.434195596089191e-01, -2.016224252678822e-01, 0.000000000000000e+00, -2.022016985161899e-01, -2.016224252678822e-01, 0.000000000000000e+00, -2.022016985161899e-01, -2.329612705983402e-01, 0.000000000000000e+00, -2.407553433813131e-01, -4.817941027244138e-02, 0.000000000000000e+00, -4.747837617622971e-02, -5.091616554758605e-02, 0.000000000000000e+00, -5.025538643137951e-02, -7.145845522619346e-02, 0.000000000000000e+00, -7.082831083005138e-02, -5.881230643750741e-02, 0.000000000000000e+00, -5.738578560343854e-02, -5.881230643750741e-02, 0.000000000000000e+00, -5.738578560343875e-02, -2.664578123169762e-02, 0.000000000000000e+00, -2.871228754985332e-02, -6.235748204565468e-01, 0.000000000000000e+00, -6.144886445509385e-01, -3.976491761078969e-01, 0.000000000000000e+00, -3.906431136892313e-01, -2.412069855310679e-01, 0.000000000000000e+00, -2.366479984560077e-01, -3.071637185480848e-01, 0.000000000000000e+00, -3.094202524419663e-01, -3.071637185480851e-01, 0.000000000000000e+00, -3.094202524419669e-01, -9.303515553038044e-02, 0.000000000000000e+00, -9.100073788082885e-02, -8.512331000042879e+01, 0.000000000000000e+00, -8.121101295791169e+01, -3.058238816573853e+01, 0.000000000000000e+00, -2.921846613328704e+01, -3.342541885939531e-01, 0.000000000000000e+00, -3.539772715920671e-01, -1.225837111816307e+01, 0.000000000000000e+00, -1.198287976465461e+01, -1.225837111816305e+01, 0.000000000000000e+00, -1.198287976465463e+01, -7.226010829939667e+02, 0.000000000000000e+00, -5.032756704734905e+02, 8.292345771610274e+05, 0.000000000000000e+00, 8.771934757308490e+05, -1.621093629312658e+05, 0.000000000000000e+00, -1.421496550330213e+05, -1.522128209383876e+01, 0.000000000000000e+00, -1.710498740545781e+01, 5.833246905352150e+04, 0.000000000000000e+00, -5.162433351612656e+04, 5.833246905352188e+04, 0.000000000000000e+00, -5.162433351612656e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.679709492234165e-04, 8.663103956746769e-04, 8.679664330839057e-04, 8.663072374607083e-04, 8.677163402154691e-04, 8.660144128537439e-04, 8.677571847679324e-04, 8.660705245521186e-04, 8.679737813194966e-04, 8.658760225953261e-04, 8.679737813194966e-04, 8.658760225953261e-04, 2.545558964452168e-03, 2.534718258804381e-03, 2.547901517933025e-03, 2.535369509240536e-03, 2.583396109718076e-03, 2.587220310250700e-03, 2.611590043262845e-03, 2.602608708942625e-03, 2.541685160467847e-03, 2.635497618015390e-03, 2.541685160467847e-03, 2.635497618015390e-03, -4.349820714998408e-03, -4.391567830846185e-03, -4.299022639595342e-03, -4.409284355223276e-03, -3.874664068589244e-03, -3.080000849386243e-03, -1.744931134254133e-03, -1.611221788235470e-03, -5.000936878759975e-03, 2.348054588757911e-03, -5.000936878759975e-03, 2.348054588757911e-03, 3.725595132114772e-02, 3.967250270451225e-02, 3.757952839385612e-02, 4.059540031315306e-02, -8.942787454224404e-03, -8.994326988358659e-03, 8.992384427346890e-03, 2.182420431031777e-02, 3.549769449751241e-02, -1.127646710578512e-02, 3.549769449751231e-02, -1.127646710578511e-02, 7.626394969203890e-03, 8.146053749021193e-03, 7.683121474709273e-03, 8.355605333360607e-03, 8.590193206079691e-03, 1.052029233717409e-02, 2.270963116150730e-03, 2.748014274953872e-03, 6.227120785541523e-03, -8.560514844213036e-04, 6.227120785541481e-03, -8.560514844213464e-04, 1.224907950564710e-03, 1.206938382201174e-03, 1.208440534910666e-03, 1.190754197241424e-03, 1.216552443837032e-03, 1.201015648532009e-03, 1.202618070130161e-03, 1.186526291829368e-03, 1.222186081787316e-03, 1.199792080356642e-03, 1.222186081787316e-03, 1.199792080356642e-03, 6.822417199136899e-03, 6.791477226013934e-03, 6.719198114082397e-03, 6.684656423330007e-03, 6.429436749020140e-03, 6.513601581022251e-03, 6.305570701623306e-03, 6.382500896704163e-03, 7.073460977911908e-03, 6.782770590360586e-03, 7.073460977911908e-03, 6.782770590360586e-03, 4.565223063415460e-02, 4.711979596697136e-02, 3.122374462876398e-02, 3.315272462786747e-02, 2.057138842384207e-02, 3.105630870819003e-02, 7.776254529371014e-03, 1.279210585976220e-02, 5.311080306447221e-02, 4.588720650245467e-02, 5.311080306447209e-02, 4.588720650245459e-02, 1.286469296896687e-02, 1.152393622583831e-02, 2.823660565903376e-02, 2.805401074517894e-02, 8.581595811326590e-03, 1.245930504975871e-02, 2.840576967479563e-03, 2.842374392742918e-03, 2.041629504345233e-02, 2.715611130021743e-02, 2.041629504345240e-02, 2.715611130021746e-02, -4.197688083062749e-03, -3.756329966014068e-03, -9.614134174202039e-04, -1.482395427260056e-03, 3.842985900854189e-03, 3.798227228942809e-03, 4.501704105321889e-03, 1.276341576070468e-02, -6.967251754943474e-03, 8.230609613374415e-03, -6.967251754943459e-03, 8.230609613374424e-03, 3.632158558646135e-01, 3.720028653278279e-01, 1.479653239476520e-01, 1.493834075385323e-01, 2.026020352784198e-01, 2.058516601483320e-01, 2.696143672871564e-01, 2.740647060670134e-01, 2.334892888742005e-01, 2.372494466633232e-01, 2.334892888742005e-01, 2.372494466633224e-01, 8.741088895590846e-02, 9.268506164094546e-02, 2.691526859726565e-02, 2.689713537050440e-02, 3.636286482343356e-02, 3.648395458798832e-02, 6.592891671767179e-02, 6.612516317792466e-02, 4.817533573152187e-02, 4.756322534712804e-02, 4.817533573152163e-02, 4.756322534712806e-02, 2.005367145996725e-02, 2.228490502606993e-02, 2.955562842496689e-02, 2.959713341764020e-02, 3.146081044479715e-02, 3.190516384749242e-02, 4.044725235609043e-02, 4.003875472085957e-02, 3.743990794763682e-02, 3.768762315883741e-02, 3.743990794763686e-02, 3.768762315883744e-02, 4.052850085949197e-02, 4.120334053178382e-02, 1.378038471308927e-02, 1.299843887463376e-02, 8.574887448116351e-03, 9.398545324202911e-03, 4.745206421011382e-02, 6.245603052987249e-02, 1.464857902393396e-02, 1.765829278496820e-02, 1.464857902393387e-02, 1.765829278496818e-02, -1.862145568083684e-03, -2.535320593662467e-03, -3.811501389568565e-03, -3.485744115556856e-03, 4.097570920473841e-03, 4.285010672258761e-03, 1.743133366290948e-02, 2.457729494325547e-02, -8.284904181889817e-03, 7.902297963307364e-03, -8.284904181889825e-03, 7.902297963307339e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05