
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc17_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.616639497485202e-05, -1.616627724289000e-05, -1.616566532800986e-05, -1.616759501178250e-05, -1.616633361163357e-05, -1.616633361163357e-05, -3.813005281604193e-03, -3.812980855913020e-03, -3.811830619447420e-03, -3.811346290831909e-03, -3.812943140460546e-03, -3.812943140460546e-03, -7.646204620425839e-02, -7.620808432030858e-02, -6.978498632839869e-02, -7.160106554591368e-02, -7.636936871000193e-02, -7.636936871000193e-02, -1.287601928184363e-03, -1.340053113393286e-03, -1.293678752849395e-01, -4.888265007539959e-04, -1.303187022114045e-03, -1.303187022114045e-03, -2.229522944974575e-07, -2.562218937673053e-07, -1.720658232838840e-05, -2.461554024966593e-08, -2.482947989036775e-07, -2.482947989036775e-07, -1.123442054549425e-03, -1.122839666649361e-03, -1.123378477315105e-03, -1.122910112847987e-03, -1.123135827030659e-03, -1.123135827030659e-03, -2.311525021315219e-02, -2.259487300074878e-02, -2.333023574603372e-02, -2.291758156523840e-02, -2.249256356108341e-02, -2.249256356108341e-02, -4.834628255115542e-02, -6.367332902054183e-02, -4.059300654834085e-02, -4.670950870636274e-02, -5.482457059978738e-02, -5.482457059978738e-02, -1.935927048498751e-04, -1.315648853120310e-03, -1.862371858135967e-04, -2.461632595573108e-02, -2.994774544428559e-04, -2.994774544428559e-04, -2.209879735726882e-08, -3.298535204479319e-08, -1.388816413729536e-08, -4.661859020287314e-05, -2.495157503862002e-08, -2.495157503862002e-08, -5.368070914946983e-02, -5.273106282133035e-02, -5.307015882779677e-02, -5.333278669614113e-02, -5.320123041311278e-02, -5.320123041311278e-02, -4.814611610340463e-02, -3.004677896478610e-02, -3.469104258185810e-02, -3.945268578375437e-02, -3.697019868274994e-02, -3.697019868274994e-02, -7.438967499330536e-02, -2.633746338000622e-03, -4.448510796725714e-03, -9.497476250061844e-03, -6.488795074861399e-03, -6.488795074861395e-03, -2.146187112267726e-02, -1.367000669605786e-05, -3.391898395538395e-05, -7.960096663344698e-03, -1.034717031360617e-04, -1.034717031360617e-04, -3.589106085828372e-07, -6.769227985537001e-10, -4.640343675511435e-09, -8.983952243064173e-05, -1.531622927864018e-08, -1.531622927864011e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc17_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc17", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.254237003327909e-09, 1.254218737247621e-09, 1.254123801009682e-09, 1.254423196225162e-09, 1.254227482832179e-09, 1.254227482832179e-09, 6.651069640580771e-05, 6.650986491727399e-05, 6.647071466570624e-05, 6.645423299268088e-05, 6.650858103427504e-05, 6.650858103427504e-05, -1.590018784787865e-01, -1.585208332876036e-01, -1.461090809605868e-01, -1.496640801772008e-01, -1.588264231031997e-01, -1.588264231031997e-01, -2.583034222903326e-03, -2.688581939185480e-03, -2.307256807553347e-01, -9.787929573428788e-04, -2.614393542049678e-03, -2.614393542049678e-03, -4.459048275913313e-07, -5.124441026519610e-07, -3.441458546941153e-05, -4.923108340777011e-08, -4.965898937279085e-07, -4.965898937279085e-07, 5.973338052464743e-06, 5.966979186759492e-06, 5.972666767390221e-06, 5.967722651379563e-06, 5.970105067632867e-06, 5.970105067632867e-06, 1.886949187526429e-03, 1.816444055202005e-03, 1.916278614688351e-03, 1.860084921127479e-03, 1.802665856246614e-03, 1.802665856246614e-03, -1.021117318272237e-01, -1.339123232332229e-01, -8.560176704811442e-02, -9.864258275419305e-02, -1.157247272269950e-01, -1.157247272269950e-01, -3.873648686637757e-04, -2.639470029856129e-03, -3.726404679591021e-04, 2.094016639651685e-03, -5.993837917332943e-04, -5.993837917332943e-04, -4.419759705864985e-08, -6.597070931214479e-08, -2.777632920041985e-08, -9.324760611811975e-05, -4.990315306562838e-08, -4.990315306562838e-08, -1.133366431686325e-01, -1.113485791979115e-01, -1.120590207106847e-01, -1.126088350727836e-01, -1.123334665484138e-01, -1.123334665484138e-01, -1.016880382680574e-01, -6.296993049715853e-02, -7.294506459358889e-02, -8.316028031647203e-02, -7.783777323421100e-02, -7.783777323421101e-02, -1.550539675782797e-01, -5.299702870172928e-03, -8.986821698200618e-03, -1.937850793175267e-02, -1.316373284754424e-02, -1.316373284754423e-02, -4.458922204205122e-02, -2.734091021004572e-05, -6.784348795135218e-05, -1.619505793021266e-02, -2.069947303474908e-04, -2.069947303474908e-04, -7.178218354836528e-07, -1.353845599306878e-09, -9.280687454380255e-09, -1.797177427286130e-04, -3.063245968329715e-08, -3.063245968329702e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05