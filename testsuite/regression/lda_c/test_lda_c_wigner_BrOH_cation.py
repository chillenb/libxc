
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_wigner_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_wigner", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.624359019547920e-02, -5.624359065852833e-02, -5.624359286171769e-02, -5.624358595705236e-02, -5.624358953801319e-02, -5.624358953801319e-02, -5.540188240846394e-02, -5.540188453897613e-02, -5.540195818475990e-02, -5.540201816935908e-02, -5.540189142189818e-02, -5.540189142189818e-02, -5.154431607063036e-02, -5.153848808552774e-02, -5.141460129614996e-02, -5.145706060654131e-02, -5.095700019370223e-02, -5.095700019370223e-02, -4.117736133765746e-02, -4.130268591782028e-02, -5.205878475568267e-02, -3.806342571892792e-02, -1.829906234274133e-02, -1.829906234274133e-02, -4.865499640850887e-03, -5.085870866806314e-03, -1.995641773917342e-02, -2.939571973693360e-03, -1.916939441411002e-03, -1.916939441411002e-03, -5.573272647868727e-02, -5.573284896209253e-02, -5.573273232886285e-02, -5.573284101504689e-02, -5.573278825021850e-02, -5.573278825021850e-02, -5.464300372085183e-02, -5.465532545526949e-02, -5.463352515725469e-02, -5.464451189147074e-02, -5.465181091455302e-02, -5.465181091455302e-02, -5.091660569649095e-02, -5.131456290895875e-02, -5.037240274136848e-02, -5.059984421443083e-02, -5.069751538089538e-02, -5.069751538089538e-02, -3.330610832409844e-02, -4.132689978337699e-02, -3.204641175568979e-02, -5.460962607531872e-02, -3.519606735801624e-02, -3.519606735801624e-02, -2.289235520424639e-03, -2.877420860727385e-03, -2.211681128765533e-03, -2.667799058481698e-02, -2.140304102554223e-03, -2.140304102554222e-03, -5.093967304782300e-02, -5.091116254109852e-02, -5.092121779309329e-02, -5.092950296345642e-02, -5.092535937248346e-02, -5.092535937248346e-02, -5.080290424310951e-02, -4.998279883483250e-02, -5.024041698353186e-02, -5.047722365809736e-02, -5.035884136483480e-02, -5.035884136483480e-02, -5.152903602892746e-02, -4.376578548002386e-02, -4.546399669485796e-02, -4.771830745405622e-02, -4.664459616390796e-02, -4.664459616390796e-02, -4.949364827388335e-02, -1.952488388681876e-02, -2.358999884974373e-02, -4.742485665952783e-02, -2.988417080854054e-02, -2.988417080854054e-02, -6.665703468231945e-03, -7.994274239495631e-04, -1.638271711935182e-03, -2.918164237943085e-02, -2.072099711660670e-03, -2.072099711660668e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_wigner_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_wigner", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.629876418379359e-02, -5.629919873666805e-02, -5.629861538575769e-02, -5.629934815521714e-02, -5.629938830974999e-02, -5.629857817793219e-02, -5.629769337098544e-02, -5.630026390533355e-02, -5.629683053552709e-02, -5.630113157358614e-02, -5.629683053552709e-02, -5.630113157358614e-02, -5.572973090742554e-02, -5.573426638015389e-02, -5.572941058397478e-02, -5.573458961866043e-02, -5.573547180808378e-02, -5.572862841967000e-02, -5.573442335682260e-02, -5.572975818533416e-02, -5.579421112561658e-02, -5.566985667590317e-02, -5.579421112561658e-02, -5.566985667590317e-02, -5.364120369464229e-02, -5.241771957662581e-02, -5.377807536296986e-02, -5.227553524513783e-02, -5.187539941957636e-02, -5.400805250184252e-02, -5.329813429716569e-02, -5.263000744709300e-02, -4.336662656578374e-02, -6.350544975905831e-02, -4.336662656578374e-02, -6.350544975905831e-02, -4.984555073560165e-02, -4.039668093075198e-02, -5.069180563890735e-02, -3.990377797704726e-02, -5.878660314980871e-02, -4.844195454147013e-02, -4.378410967957113e-02, -4.065301436000429e-02, -4.655740982768717e-03, -1.203426365272963e-01, -4.655740982768712e-03, -1.203426365272963e-01, -7.318716878568495e-03, -5.535827579237866e-03, -7.793120367788389e-03, -5.677854376804937e-03, -2.778435715386326e-02, -2.122924627726223e-02, -3.723825133298740e-03, -4.020318200435201e-03, -9.509208167208812e-04, -1.099991824496583e-02, -9.509208167208817e-04, -1.099991824496583e-02, -5.591581406541105e-02, -5.599592446176981e-02, -5.591484053868205e-02, -5.599706454661313e-02, -5.591507301505884e-02, -5.599667430943852e-02, -5.591575966375275e-02, -5.599613357104502e-02, -5.591512202147104e-02, -5.599670074650312e-02, -5.591512202147104e-02, -5.599670074650312e-02, -5.520534954604144e-02, -5.522191707479002e-02, -5.517749105709616e-02, -5.526674694784605e-02, -5.549096831531556e-02, -5.492449353252660e-02, -5.551045183079358e-02, -5.492022407103781e-02, -5.450404239092915e-02, -5.594327815766323e-02, -5.450404239092915e-02, -5.594327815766323e-02, -5.216590765926180e-02, -5.297579851772862e-02, -5.292960329519544e-02, -5.278986023921960e-02, -5.788796429120749e-02, -4.695472553058051e-02, -5.751397961001065e-02, -4.757446901376694e-02, -4.511124596572276e-02, -6.072236206437995e-02, -4.511124596572277e-02, -6.072236206437993e-02, -3.881382914661546e-02, -3.691693377864125e-02, -4.563158622230754e-02, -4.439714984801855e-02, -4.207072800493482e-02, -3.196610629666464e-02, -5.511709941534831e-02, -5.526433846058749e-02, -4.619754043705346e-02, -3.396410652930656e-02, -4.619754043705346e-02, -3.396410652930656e-02, -3.301462271691833e-03, -2.771584744645437e-03, -3.920248681161267e-03, -3.660819285785689e-03, -3.335070230874473e-03, -2.570157446551935e-03, -3.185984550459389e-02, -3.087859432416158e-02, -6.223348546722610e-03, -1.506101931514441e-03, -6.223348546722612e-03, -1.506101931514441e-03, -5.317887842225564e-02, -5.199963464831976e-02, -5.316367771806118e-02, -5.197323967559721e-02, -5.317094071919507e-02, -5.198068625286118e-02, -5.317161114934364e-02, -5.199203022558923e-02, -5.317121512124563e-02, -5.198641648334303e-02, -5.317121512124563e-02, -5.198641648334303e-02, -5.299133693424963e-02, -5.198534684355373e-02, -5.253298859115690e-02, -5.123651089873767e-02, -5.274330114108711e-02, -5.140842764755128e-02, -5.278083339323766e-02, -5.171772138433086e-02, -5.272241989197943e-02, -5.160206610697660e-02, -5.272241989197943e-02, -5.160206610697660e-02, -5.326753751599294e-02, -5.276414003500204e-02, -4.779963473911141e-02, -4.628349319471718e-02, -4.978368327437613e-02, -4.706103368906404e-02, -5.096419414203010e-02, -4.938540694053340e-02, -4.928692726187529e-02, -4.938567250581784e-02, -4.928692726187531e-02, -4.938567250581782e-02, -5.254447650411089e-02, -5.050642736536749e-02, -2.418380384262179e-02, -2.338491277027202e-02, -3.060616896703368e-02, -2.593812959097636e-02, -5.211412562837063e-02, -4.785391215569838e-02, -3.982966723587900e-02, -3.003405722511956e-02, -3.982966723587899e-02, -3.003405722511957e-02, -9.375889161411164e-03, -7.948876660470422e-03, -1.067620757022323e-03, -1.056670303958333e-03, -2.501143460401037e-03, -1.891773420477816e-03, -3.532908903877335e-02, -3.248779632393757e-02, -5.523224679448937e-03, -1.543374025641345e-03, -5.523224679448931e-03, -1.543374025641344e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05