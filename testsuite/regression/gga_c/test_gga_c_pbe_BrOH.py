
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.096054000209373e-02, -5.096127809985089e-02, -5.096350446608085e-02, -5.095144013365942e-02, -5.096093816320627e-02, -5.096093816320627e-02, -4.137436897843448e-02, -4.137852876403897e-02, -4.149305563811514e-02, -4.129611188446786e-02, -4.137651223260846e-02, -4.137651223260846e-02, -2.588385970831824e-02, -2.566413995104777e-02, -2.002307094188837e-02, -2.031128893025791e-02, -2.580462486770759e-02, -2.580462486770759e-02, -8.879441230614644e-03, -9.563629767964408e-03, -3.285860114837120e-02, -2.374241238048071e-03, -9.090019117586504e-03, -9.090019117586504e-03, -1.241343167082831e-07, -1.534593247651874e-07, -3.149927341290885e-05, -3.057245239894329e-09, -1.520771019066711e-07, -1.520771019066711e-07, -5.861812257157630e-02, -5.882299950810805e-02, -5.863855317919035e-02, -5.879785475393116e-02, -5.872372138422077e-02, -5.872372138422077e-02, -2.046142860693844e-02, -2.090909887699305e-02, -1.984079160888845e-02, -2.018483054226389e-02, -2.159997189329698e-02, -2.159997189329698e-02, -3.896589775802473e-02, -5.635546565502639e-02, -3.753010990615920e-02, -5.102185741482872e-02, -4.144374211638558e-02, -4.144374211638558e-02, -6.140141495143812e-04, -4.240213483146681e-03, -6.764414740706327e-04, -7.183609218399736e-02, -1.348222919174980e-03, -1.348222919174980e-03, -2.746320925563308e-09, -5.099873152549153e-09, -3.842130885744162e-09, -1.339319808191252e-04, -4.903228519172304e-09, -4.903228519172304e-09, -6.149874138674653e-02, -5.637786621437589e-02, -5.812356742191978e-02, -5.954005715822859e-02, -5.882429001826525e-02, -5.882429001826525e-02, -6.169392785522880e-02, -2.895693947838382e-02, -3.627190289694263e-02, -4.454271362771697e-02, -4.022656111672120e-02, -4.022656111672120e-02, -5.636290611811082e-02, -7.292512196097774e-03, -1.205816121489509e-02, -2.440886282339105e-02, -1.769622867098732e-02, -1.769622867098731e-02, -2.731344076214311e-02, -1.892497486065396e-05, -7.133835160738992e-05, -2.927730220761547e-02, -4.272675563259798e-04, -4.272675563259798e-04, -1.904931004124213e-07, -2.005383976925495e-11, -3.600759229185977e-10, -4.329186953827006e-04, -3.369326359045795e-09, -3.369326356908343e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.271535083987370e-01, -1.271544122396350e-01, -1.271571475421094e-01, -1.271423731970662e-01, -1.271539958942170e-01, -1.271539958942170e-01, -1.091199757986678e-01, -1.091249470798903e-01, -1.092618323984621e-01, -1.090274316655483e-01, -1.091225628112825e-01, -1.091225628112825e-01, -7.647310201349550e-02, -7.617524167844208e-02, -6.713437385117585e-02, -6.770891271990619e-02, -7.636607970473057e-02, -7.636607970473057e-02, -3.580396611781717e-02, -3.765016670652626e-02, -8.620876561872462e-02, -1.236742458668383e-02, -3.638185767617162e-02, -3.638185767617162e-02, -8.055249313537823e-07, -9.952240093169411e-07, -1.975529471789978e-04, -1.998851945683567e-08, -9.863538418429856e-07, -9.863538418429856e-07, -1.293588729959759e-01, -1.295362623670392e-01, -1.293766076547762e-01, -1.295145435681006e-01, -1.294503947759391e-01, -1.294503947759391e-01, -7.289498286910215e-02, -7.389393094813154e-02, -7.150207478596370e-02, -7.228905282014769e-02, -7.537630402258852e-02, -7.537630402258852e-02, -8.420134402250117e-02, -8.277480808217015e-02, -8.264663327600570e-02, -8.231884304913582e-02, -8.556338113929791e-02, -8.556338113929791e-02, -3.560037702060391e-03, -2.052514172046345e-02, -3.900444773327664e-03, -1.178686899785536e-01, -7.420682734537302e-03, -7.420682734537302e-03, -1.796067741088308e-08, -3.331585414639346e-08, -2.515234289172461e-08, -8.190712454595034e-04, -3.205319185875183e-08, -3.205319185567008e-08, -7.565909762125839e-02, -8.031451838311958e-02, -7.891998485491258e-02, -7.764365834737645e-02, -7.830498235506023e-02, -7.830498235506023e-02, -7.371766943695926e-02, -7.630352303908150e-02, -8.123501549200068e-02, -8.281690147025297e-02, -8.247509044446295e-02, -8.247509044446294e-02, -8.474065310405153e-02, -3.183582608424547e-02, -4.564723921594136e-02, -6.709481535921981e-02, -5.746942048382449e-02, -5.746942048382447e-02, -7.338975554202548e-02, -1.193504511796647e-04, -4.419071275331208e-04, -6.916721170625963e-02, -2.517035712228023e-03, -2.517035712228037e-03, -1.233952771233740e-06, -1.321314906269836e-10, -2.363662832265945e-09, -2.547828080320751e-03, -2.205293940126641e-08, -2.205293939753708e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.927907276572664e-10, 1.927937741909165e-10, 1.927978782024764e-10, 1.927481892573838e-10, 1.927924135892734e-10, 1.927924135892734e-10, 1.079462704051496e-06, 1.079639676721703e-06, 1.084298901378030e-06, 1.075169227678776e-06, 1.079531177681063e-06, 1.079531177681063e-06, 1.699382159972962e-03, 1.686258189006026e-03, 1.331924631218257e-03, 1.309005286321670e-03, 1.694690511897008e-03, 1.694690511897008e-03, 2.070295992081943e-01, 2.156425041190170e-01, 1.011222168920672e-03, 1.494366038387610e-01, 2.099140496679336e-01, 2.099140496679336e-01, 2.948719632297825e-02, 3.177314056001308e-02, 6.521626262449823e-02, 5.521155581012880e-03, 3.311676184149787e-02, 3.311676184149787e-02, 2.837528188730227e-07, 2.855702782735333e-07, 2.839322031773686e-07, 2.853452845182437e-07, 2.846902375971897e-07, 2.846902375971897e-07, 6.064470018048351e-06, 6.051789517287548e-06, 5.872614881238095e-06, 5.864739915989260e-06, 6.311753492168359e-06, 6.311753492168359e-06, 6.263364071402385e-03, 7.960889481226221e-03, 7.605069825253479e-03, 1.036266616253009e-02, 5.767579033770192e-03, 5.767579033770192e-03, 9.365090901979554e-02, 6.851098986781592e-02, 1.154461858454678e-01, 6.082235103818654e-05, 1.458908270564254e-01, 1.458908270564254e-01, 5.689933158712631e-03, 7.087435648747788e-03, 2.145382733561061e-02, 1.051865276831310e-01, 1.094795505527167e-02, 1.094795505555073e-02, 1.206896525695809e-02, 1.045512829025688e-02, 1.098054816859489e-02, 1.142542866312120e-02, 1.119855225654212e-02, 1.119855225654212e-02, 1.419065707346607e-02, 7.942209324388206e-03, 9.065171798902092e-03, 1.046485509881154e-02, 9.732870152657853e-03, 9.732870152657854e-03, 6.343535576599726e-03, 4.862475030186091e-02, 4.306640570804586e-02, 3.556898361501168e-02, 4.082231596068073e-02, 4.082231596068074e-02, 1.202466809549191e-02, 4.496675758227553e-02, 7.012009044295377e-02, 6.088899082996579e-02, 1.556943086605904e-01, 1.556943086605898e-01, 2.438024817906253e-02, 1.779839569319765e-03, 4.326607523398382e-03, 2.013311254719649e-01, 1.480927522127662e-02, 1.480927522044869e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05