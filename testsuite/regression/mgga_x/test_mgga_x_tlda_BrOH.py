
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tlda_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tlda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.721988214512705e+01, -1.721983104253812e+01, -1.721964242649474e+01, -1.722046883096336e+01, -1.721985498917750e+01, -1.721985498917750e+01, -3.477895473159631e+00, -3.477742384049245e+00, -3.473261705638000e+00, -3.475274457786352e+00, -3.477869520003178e+00, -3.477869520003178e+00, -7.799589188516559e-01, -7.800515984370620e-01, -7.820264041430152e-01, -7.779619415510433e-01, -7.799768217395332e-01, -7.799768217395332e-01, -2.206747480269466e-01, -2.214736518677956e-01, -1.013203080006705e+00, -2.039843347579743e-01, -2.209480059081602e-01, -2.209480059081602e-01, -3.826094961955141e-02, -3.936743570433468e-02, -1.007493954739250e-01, -2.733511614984147e-02, -3.894547245361216e-02, -3.894547245361216e-02, -3.890628176934085e+00, -3.885242753146168e+00, -3.890030042540999e+00, -3.885846186312114e+00, -3.887934472448846e+00, -3.887934472448846e+00, -2.041283654297516e+00, -2.039187988584145e+00, -2.042733808261123e+00, -2.040857205582868e+00, -2.037819604350109e+00, -2.037819604350109e+00, -5.308852812587008e-01, -5.095308127373863e-01, -5.298931621679163e-01, -5.093494856137808e-01, -5.303686086878396e-01, -5.303686086878396e-01, -1.693734302740723e-01, -2.473802163420677e-01, -1.653946878153693e-01, -1.680692677814041e+00, -1.772923048686711e-01, -1.772923048686711e-01, -2.564635066360471e-02, -2.831501411760043e-02, -2.076969955686738e-02, -1.268011852483281e-01, -2.622478410701286e-02, -2.622478410701286e-02, -3.368081897335887e-01, -3.844563517918350e-01, -3.702762951944302e-01, -3.574220539916348e-01, -3.641037515632469e-01, -3.641037515632469e-01, -3.463784694974105e-01, -4.666121792846756e-01, -4.521652321086508e-01, -4.293151487855146e-01, -4.425289623046293e-01, -4.425289623046293e-01, -5.561456097656284e-01, -2.860383268777037e-01, -3.171602180062186e-01, -3.643745874637737e-01, -3.367376474639166e-01, -3.367376474639165e-01, -4.479367154110673e-01, -9.694208413584489e-02, -1.174461778765982e-01, -3.291711244374653e-01, -1.416345081127457e-01, -1.416345081127457e-01, -4.623645308320139e-02, -1.288686061013006e-02, -1.770019310377888e-02, -1.352053386384191e-01, -2.299010038311694e-02, -2.299010038311693e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tlda_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tlda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.721988214512705e+01, -1.721983104253812e+01, -1.721964242649474e+01, -1.722046883096336e+01, -1.721985498917750e+01, -1.721985498917750e+01, -3.477895473159631e+00, -3.477742384049245e+00, -3.473261705638000e+00, -3.475274457786352e+00, -3.477869520003178e+00, -3.477869520003178e+00, -7.799589188516559e-01, -7.800515984370620e-01, -7.820264041430152e-01, -7.779619415510433e-01, -7.799768217395332e-01, -7.799768217395332e-01, -2.206747480269466e-01, -2.214736518677956e-01, -1.013203080006705e+00, -2.039843347579743e-01, -2.209480059081602e-01, -2.209480059081602e-01, -3.826094961955141e-02, -3.936743570433468e-02, -1.007493954739250e-01, -2.733511614984147e-02, -3.894547245361216e-02, -3.894547245361216e-02, -3.890628176934085e+00, -3.885242753146168e+00, -3.890030042540999e+00, -3.885846186312114e+00, -3.887934472448846e+00, -3.887934472448846e+00, -2.041283654297516e+00, -2.039187988584145e+00, -2.042733808261123e+00, -2.040857205582868e+00, -2.037819604350109e+00, -2.037819604350109e+00, -5.308852812587008e-01, -5.095308127373863e-01, -5.298931621679163e-01, -5.093494856137808e-01, -5.303686086878396e-01, -5.303686086878396e-01, -1.693734302740723e-01, -2.473802163420677e-01, -1.653946878153693e-01, -1.680692677814041e+00, -1.772923048686711e-01, -1.772923048686711e-01, -2.564635066360471e-02, -2.831501411760043e-02, -2.076969955686738e-02, -1.268011852483281e-01, -2.622478410701286e-02, -2.622478410701286e-02, -3.368081897335887e-01, -3.844563517918350e-01, -3.702762951944302e-01, -3.574220539916348e-01, -3.641037515632469e-01, -3.641037515632469e-01, -3.463784694974105e-01, -4.666121792846757e-01, -4.521652321086508e-01, -4.293151487855146e-01, -4.425289623046293e-01, -4.425289623046293e-01, -5.561456097656284e-01, -2.860383268777037e-01, -3.171602180062186e-01, -3.643745874637737e-01, -3.367376474639166e-01, -3.367376474639165e-01, -4.479367154110673e-01, -9.694208413584489e-02, -1.174461778765982e-01, -3.291711244374653e-01, -1.416345081127457e-01, -1.416345081127457e-01, -4.623645308320139e-02, -1.288686061013006e-02, -1.770019310377888e-02, -1.352053386384191e-01, -2.299010038311694e-02, -2.299010038311693e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tlda_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tlda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tlda_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tlda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tlda_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tlda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.365398740923181e-03, -3.365463197650406e-03, -3.365738042751156e-03, -3.364690397243604e-03, -3.365432743367127e-03, -3.365432743367127e-03, -8.727637606303571e-03, -8.729229418887282e-03, -8.776963508570282e-03, -8.757741661619795e-03, -8.728037928364419e-03, -8.728037928364419e-03, -2.430285855693753e-02, -2.421658247257418e-02, -2.210813369153519e-02, -2.311115999416554e-02, -2.427334370373602e-02, -2.427334370373602e-02, -7.764814983177024e-02, -7.963186882157763e-02, -1.469159902915497e-02, -4.052913274967482e-02, -7.819422539757245e-02, -7.819422539757245e-02, -1.496935370615517e-02, -1.534905857190009e-02, -2.402721892557962e-02, -6.343664034511841e-03, -1.552936673187642e-02, -1.552936673187642e-02, -1.868213329670152e-02, -1.879595687991571e-02, -1.869467863538213e-02, -1.878311209050979e-02, -1.873904268958857e-02, -1.873904268958857e-02, -1.306625747807534e-02, -1.339933408192428e-02, -1.291817936202316e-02, -1.318145140595985e-02, -1.349192218161647e-02, -1.349192218161647e-02, -7.471692165486779e-02, -1.128571652755569e-01, -6.431765939415973e-02, -8.548830922401263e-02, -8.399087430178560e-02, -8.399087430178560e-02, -3.381578806518541e-02, -5.023229435222942e-02, -3.577726313363932e-02, -2.682804426736659e-02, -4.355083224931198e-02, -4.355083224931198e-02, -7.349889962711529e-03, -7.383605301266448e-03, -1.073837342757925e-02, -2.594079545780303e-02, -7.590401955562072e-03, -7.590401955562066e-03, -5.067236934507033e-01, -2.937239621977035e-01, -3.433430976943803e-01, -3.972239550241968e-01, -3.680401205148629e-01, -3.680401205148629e-01, -4.107692493063322e-01, -8.142898405915136e-02, -1.052436858305614e-01, -1.454833969365331e-01, -1.215198270703768e-01, -1.215198270703769e-01, -9.165600804086037e-02, -5.591444825105601e-02, -6.197072753647017e-02, -7.432788626939109e-02, -7.050337316117662e-02, -7.050337316117661e-02, -7.035026439586511e-02, -2.226923883184136e-02, -2.564667947212536e-02, -9.413072198475252e-02, -3.697807730514564e-02, -3.697807730514565e-02, -1.129957351050069e-02, -3.531548409945847e-03, -6.802251224773687e-03, -3.866513108115845e-02, -7.888653942035688e-03, -7.888653942035676e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05