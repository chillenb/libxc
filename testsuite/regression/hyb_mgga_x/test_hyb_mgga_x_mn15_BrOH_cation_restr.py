
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mn15_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.332785775530794e+00, -2.332156418428802e+00, -2.329863283755326e+00, -2.339186637271108e+00, -2.334230792622413e+00, -2.334230792622413e+00, -2.450804683429125e+00, -2.450449902167035e+00, -2.441135786636790e+00, -2.445188089351688e+00, -2.444466072149530e+00, -2.444466072149530e+00, -5.761187316089391e-01, -5.775281910661428e-01, -6.151970668037005e-01, -6.067781283438779e-01, -6.076319654280476e-01, -6.076319654280476e-01, -2.003640076551245e-01, -1.995514729781684e-01, -8.043442824710607e-01, -1.403600948076194e-01, -1.696723960005823e-01, -1.696723960005825e-01, 7.951102521984863e-03, 8.311061782685406e-03, 1.749713455235467e-02, 4.838973404794355e-03, 5.971368020365760e-03, 5.971368020365768e-03, 5.599719864443164e-01, 5.985448978283751e-01, 5.619540485797446e-01, 5.959901285476468e-01, 5.793804928291069e-01, 5.793804928291069e-01, -1.468541803490446e+00, -1.452050948812260e+00, -1.480428766945886e+00, -1.464513705753251e+00, -1.453618176648770e+00, -1.453618176648770e+00, -3.157073367080151e-01, -2.187471655698778e-01, -3.265192856491457e-01, -2.799328583086874e-01, -3.070812478944065e-01, -3.070812478944065e-01, -6.260250248336058e-02, -2.184287773923171e-01, -4.920647032721181e-02, -7.291457090846017e-01, -1.021024533830817e-01, -1.021024533830818e-01, 3.791694184446569e-03, 4.738090913648795e-03, 3.672709681936668e-03, -3.809298796106031e-03, 4.382686135241167e-03, 4.382686135241172e-03, 8.690620146857782e-02, -5.165018291648765e-02, -9.772726468495240e-03, 3.017627924992047e-02, 9.534175977720660e-03, 9.534175977720660e-03, 8.074983338377233e-02, -3.016558095747148e-01, -2.714606110433954e-01, -2.138719246082686e-01, -2.492881935484515e-01, -2.492881935484515e-01, -2.626530730703344e-01, -2.673139588491172e-01, -2.761151459791280e-01, -2.484047105882372e-01, -2.602905310777019e-01, -2.602905310777019e-01, -2.963951461715233e-01, 1.806826500420384e-02, 1.018154778875303e-02, -2.085379187985924e-01, -3.201583038578514e-02, -3.201583038578513e-02, 1.066164437408961e-02, 1.326848429608953e-03, 2.733652288069024e-03, -2.553438470745239e-02, 4.072625675437578e-03, 4.072625675437584e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mn15_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.866540727166606e+01, 3.866719104341050e+01, 3.867419439246584e+01, 3.864776989277989e+01, 3.866174569927219e+01, 3.866174569927219e+01, -1.002798735984343e+00, -1.002083292275944e+00, -9.835617781952627e-01, -9.957188216111656e-01, -9.924140267885091e-01, -9.924140267885091e-01, -2.827465097880774e-01, -2.816041725698408e-01, -2.506781858033513e-01, -2.416085073614108e-01, -2.456979026648111e-01, -2.456979026648111e-01, -1.018137382898146e-01, -9.292416529327539e-02, -5.330199708577680e-01, -1.590984208454599e-01, -1.488290898783551e-01, -1.488290898783555e-01, 1.020877278616689e-02, 1.064040472606620e-02, 8.924725407030270e-03, 6.333420319875427e-03, 7.764636147412643e-03, 7.764636147412662e-03, 1.073570142591314e+01, 1.080574022466699e+01, 1.073963106949999e+01, 1.080142453535399e+01, 1.077086957110940e+01, 1.077086957110940e+01, -9.634904829665702e-01, -9.870138939263926e-01, -9.544171838853858e-01, -9.809301972521856e-01, -9.775958450301134e-01, -9.775958450301134e-01, -1.790364425330397e-01, 6.154366120929954e-02, -2.036819760431688e-01, -1.657022504694421e-01, -1.579573296686216e-01, -1.579573296686216e-01, -1.131028830901128e-01, -1.716542741332142e-01, -9.791379054620429e-02, 7.747074095138988e-01, -1.512057043011511e-01, -1.512057043011504e-01, 4.987654783075075e-03, 6.205025209372273e-03, 4.821949964568889e-03, -3.464049095695231e-02, 5.741069255436534e-03, 5.741069255436521e-03, 3.836386489828388e-01, 3.160680862197842e-01, 3.477631048087563e-01, 3.684458048555728e-01, 3.588507675045868e-01, 3.588507675045868e-01, 3.588159099720061e-01, -1.692876292402441e-01, -1.404557721600823e-01, 1.333009154921134e-02, -8.335778477357016e-02, -8.335778477356996e-02, -5.503207978332359e-03, -1.233827170149992e-01, -6.951769019339027e-02, -9.373689325982074e-02, -5.623258754272366e-02, -5.623258754272396e-02, -1.448228284934338e-01, 1.062111862056547e-02, -8.214072096125609e-03, -1.009258667334174e-01, -7.996599171264837e-02, -7.996599171264815e-02, 1.344865722733600e-02, 1.761646833775506e-03, 3.609094513910018e-03, -7.216277126143719e-02, 5.340754720371205e-03, 5.340754720371123e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.704283802221269e-08, -4.704450051752924e-08, -4.705084684633238e-08, -4.702628018821363e-08, -4.703929819159769e-08, -4.703929819159769e-08, -2.966409852280053e-07, -3.011620446889248e-07, -4.203341097557177e-07, -3.584263068953707e-07, -3.709069941688685e-07, -3.709069941688685e-07, -7.971970528790074e-03, -8.016383311817244e-03, -9.045524094880193e-03, -8.297304526118267e-03, -8.504200404919967e-03, -8.504200404919967e-03, -1.796011486135654e+00, -1.787167642667127e+00, -5.123906043804305e-03, -2.361132842959033e+00, -2.310929031704184e+00, -2.310929031704181e+00, 1.039898032085898e+00, 1.043311122461365e+00, -1.756170470349510e+00, 1.336285852232928e+00, 1.522703397159376e+00, 1.522703397158242e+00, -1.887784026896048e-05, -1.901893290828708e-05, -1.888611350072371e-05, -1.901057068893571e-05, -1.894836415722216e-05, -1.894836415722216e-05, 2.112323900378275e-05, 1.954837620355499e-05, 2.324106295141768e-05, 2.233295762599086e-05, 1.901423548207980e-05, 1.901423548207980e-05, -1.270191078547055e-02, -2.211038851194737e-02, -1.502562552785467e-02, -1.590082478215881e-02, -1.366708338889857e-02, -1.366708338889857e-02, -2.519100453664103e+00, -1.144386930604990e+00, -2.672403229912370e+00, -2.330634468874801e-04, -3.057912487807352e+00, -3.057912487807352e+00, 1.925122522820613e+00, 1.587820685821652e+00, 1.090844109647857e+01, -3.063263831197448e+00, 4.789995382695087e+00, 4.789995382755442e+00, -9.589053185773518e-02, -7.319653123024254e-02, -8.045202026056129e-02, -8.703377609190366e-02, -8.366893234567598e-02, -8.366893234567598e-02, -9.790102473756382e-02, -1.891159170726713e-02, -2.492251335044006e-02, -4.254579373474134e-02, -3.176542335896534e-02, -3.176542335896532e-02, -1.454253204382408e-02, -6.227937984621731e-01, -3.689857017536403e-01, -1.258658843698643e-01, -2.240864160415953e-01, -2.240864160415955e-01, -2.933060280228667e-02, -1.532873859470304e+00, -2.144908009302772e+00, -1.372292552951523e-01, -3.808565878334041e+00, -3.808565878334073e+00, 5.113522075973417e-01, 1.005080642620442e+01, 4.422746257163801e+00, -3.757755048675518e+00, 6.207893436061052e+00, 6.207893436152905e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.346485262004345e-02, -1.346552620172067e-02, -1.346786306268103e-02, -1.345788363255930e-02, -1.346320495884796e-02, -1.346320495884796e-02, -1.717030263994618e-02, -1.717431385074699e-02, -1.727774767371656e-02, -1.720491398316398e-02, -1.722656624927479e-02, -1.722656624927479e-02, -2.842150240536434e-02, -2.833307021496930e-02, -2.632492896627752e-02, -2.872196629724005e-02, -2.802735331784996e-02, -2.802735331784996e-02, 6.369547994389360e-02, 5.901224865157786e-02, -1.412312915966678e-02, 7.548825928123830e-02, 8.762224204897674e-02, 8.762224204897717e-02, 6.383238657027058e-05, 7.741459955446060e-05, 5.133931626261438e-03, 9.154610696405159e-06, 2.515632734386829e-05, 2.515632734386828e-05, -7.195922784728860e-02, -7.271276391453231e-02, -7.199679423450679e-02, -7.266174121824950e-02, -7.233887872170731e-02, -7.233887872170731e-02, -2.287747413144127e-02, -2.237823421056335e-02, -2.326131479072324e-02, -2.280930959927428e-02, -2.244346033724053e-02, -2.244346033724053e-02, -6.529393192848985e-02, -1.914526785497900e-01, -5.389017952587159e-02, -8.262473980694618e-02, -7.248332736935396e-02, -7.248332736935396e-02, 4.159691009205661e-02, 6.190285764333808e-02, 3.717736240026504e-02, -7.336654873042436e-02, 7.774543996734085e-02, 7.774543996734037e-02, 2.923321979209555e-06, 7.563781490377224e-06, 2.745558142796730e-05, 1.966987543517618e-02, 1.836911379711583e-05, 1.836911379712155e-05, -9.069253349483480e-01, -5.447749057604732e-01, -6.506979832231371e-01, -7.547646463288022e-01, -7.006469568646778e-01, -7.006469568646778e-01, -9.258095850323310e-01, -6.174163248526020e-02, -7.245866058981432e-02, -1.696503774548330e-01, -1.019358432512965e-01, -1.019358432512969e-01, -1.414549863651713e-01, 2.691505747661748e-02, -1.233572231091165e-02, -5.401527033738462e-02, -4.460508276592812e-02, -4.460508276592807e-02, -6.300118495846453e-02, 4.628285610445024e-03, 9.320484345183698e-03, -7.279642532453351e-02, 4.359780504165668e-02, 4.359780504165697e-02, 8.452966061758565e-05, 1.793330295892671e-07, 4.729157945806249e-06, 4.125890939449490e-02, 2.076752267958771e-05, 2.076752267961878e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05