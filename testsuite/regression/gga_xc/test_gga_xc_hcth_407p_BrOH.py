
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_407p_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.121048784214809e+01, -2.121054098807018e+01, -2.121081309893493e+01, -2.120994213440677e+01, -2.121051557679399e+01, -2.121051557679399e+01, -3.518244393372704e+00, -3.518245852523377e+00, -3.518429325771963e+00, -3.518846754488546e+00, -3.518259858469788e+00, -3.518259858469788e+00, -7.276366760598959e-01, -7.274563486170658e-01, -7.264739697096756e-01, -7.307485054549527e-01, -7.275676929040230e-01, -7.275676929040230e-01, -2.484612992400260e-01, -2.487286767708108e-01, -8.502615435520245e-01, -2.116479703486435e-01, -2.485039779255419e-01, -2.485039779255419e-01, 7.248246697561895e-03, 7.285608027348645e-03, -2.746142245234667e-02, 5.188808879976021e-03, 7.264916506854716e-03, 7.264916506854716e-03, -5.231109723685016e+00, -5.232231777410829e+00, -5.231226786261102e+00, -5.232099172985345e+00, -5.231681067106532e+00, -5.231681067106532e+00, -2.083168469399077e+00, -2.093183538145792e+00, -2.083518125818519e+00, -2.091177914359158e+00, -2.089965301522162e+00, -2.089965301522162e+00, -6.198218811558012e-01, -6.715887897683682e-01, -5.905347455021120e-01, -6.112281724862014e-01, -6.409501898595718e-01, -6.409501898595718e-01, -1.502295303237834e-01, -2.771818987693163e-01, -1.490796996700496e-01, -1.972098047574737e+00, -1.817277712205140e-01, -1.817277712205140e-01, 5.063497369071326e-03, 5.523705894969523e-03, 4.516004324416150e-03, -6.704358394968099e-02, 5.186638921633020e-03, 5.186638921633016e-03, -6.459525356841833e-01, -6.371243723148484e-01, -6.403518221787428e-01, -6.428927809003411e-01, -6.416300662716526e-01, -6.416300662716528e-01, -6.268697171173286e-01, -5.512865318652478e-01, -5.654279160474285e-01, -5.816399289579707e-01, -5.727630864809036e-01, -5.727630864809035e-01, -7.020272748024161e-01, -3.195264651171239e-01, -3.471916134391295e-01, -3.982078965954300e-01, -3.686890942154962e-01, -3.686890942154962e-01, -5.023190378006026e-01, -1.999971653224269e-02, -5.008017114522446e-02, -3.703499004950350e-01, -1.138172148282093e-01, -1.138172148282092e-01, 7.411137427988713e-03, 1.997336943335625e-03, 3.435572370931915e-03, -1.078249106269943e-01, 4.630694604232715e-03, 4.630694604232760e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_407p_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.844425738823362e+01, -2.844435121277462e+01, -2.844478460348342e+01, -2.844324783201788e+01, -2.844430674430328e+01, -2.844430674430328e+01, -4.575723778081018e+00, -4.575783312463871e+00, -4.577604219908828e+00, -4.575429236242798e+00, -4.575773818783102e+00, -4.575773818783102e+00, -8.373229858970108e-01, -8.350678327776010e-01, -7.684408103345042e-01, -7.770572916242454e-01, -8.365087859294270e-01, -8.365087859294270e-01, -2.117475765953321e-01, -2.146895461072791e-01, -1.040427575004581e+00, -2.325270763910822e-01, -2.125749529164470e-01, -2.125749529164470e-01, 6.294663131639197e-03, 5.991629884667242e-03, -8.816700835451806e-02, 6.198498802133534e-03, 6.011135723791654e-03, 6.011135723791654e-03, -7.023107103176174e+00, -7.025705088788476e+00, -7.023373940758011e+00, -7.025394068891162e+00, -7.024438951711208e+00, -7.024438951711208e+00, -2.256631143095337e+00, -2.286275669574754e+00, -2.230259348532194e+00, -2.253662337136611e+00, -2.310082945557306e+00, -2.310082945557306e+00, -7.828895846224814e-01, -9.085573520216125e-01, -7.401052543029735e-01, -8.205076324254482e-01, -8.199195067517215e-01, -8.199195067517215e-01, -2.601392843801758e-01, -2.375900498372475e-01, -2.545443054982764e-01, -2.662297390566886e+00, -2.491894593764666e-01, -2.491894593764666e-01, 6.074082045383706e-03, 6.489732369150601e-03, 5.441194772293892e-03, -1.671677167454627e-01, 6.143061225232047e-03, 6.143061225232465e-03, -8.572354002386621e-01, -8.615300328659705e-01, -8.622182470855219e-01, -8.605111802882537e-01, -8.615819582039922e-01, -8.615819582039923e-01, -8.299374274871960e-01, -6.515459106422581e-01, -7.035010260998038e-01, -7.580081338755053e-01, -7.289586594764258e-01, -7.289586594764254e-01, -9.494188756128378e-01, -2.554222049281042e-01, -3.086281452476745e-01, -4.489835628788926e-01, -3.748177975995778e-01, -3.748177975995786e-01, -5.844470608264425e-01, -6.987894833217775e-02, -1.356521314891311e-01, -4.385321834000777e-01, -2.258935973650906e-01, -2.258935973650902e-01, 5.486784024929650e-03, 2.579433268332165e-03, 4.314941454121478e-03, -2.172459909534954e-01, 5.581214253281863e-03, 5.581214253281860e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_407p_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.940484840052004e-10, 4.941129018374202e-10, 4.942975319255383e-10, 4.932446630356276e-10, 4.940833147147517e-10, 4.940833147147517e-10, -1.503745305730180e-06, -1.503072378605549e-06, -1.484285285931225e-06, -1.515176594069941e-06, -1.503369920851913e-06, -1.503369920851913e-06, -4.066491953117249e-03, -4.107080839227337e-03, -5.291083720592512e-03, -5.088278222220164e-03, -4.081156554052911e-03, -4.081156554052911e-03, -8.784823941245499e-01, -8.567317476032907e-01, -1.548027254948133e-03, -7.422381772999485e-01, -8.726079535264669e-01, -8.726079535264669e-01, 7.796678171565488e+01, 7.736670885572421e+01, 1.701360762194766e+01, 6.247063938745681e+01, 8.053495686753195e+01, 8.053495686753195e+01, 2.788967974146267e-07, 2.852315753817931e-07, 2.795271384628266e-07, 2.844526633321168e-07, 2.821625961661230e-07, 2.821625961661230e-07, -5.756369362739110e-05, -5.500006328265822e-05, -5.959738466181252e-05, -5.753856077230815e-05, -5.325492659995280e-05, -5.325492659995280e-05, -4.907470852938176e-03, 8.462500354703126e-03, -6.635037503279975e-03, 5.458306349732809e-03, -3.469272065792818e-03, -3.469272065792818e-03, 1.680785959610810e+00, -5.400825048757518e-01, 1.772412518045196e+00, 6.875672660794423e-05, 9.344528762849542e-02, 9.344528762849542e-02, 6.655997404884798e+01, 6.557088069962334e+01, 1.940499138169246e+02, 1.058699468272042e+01, 9.795494558944391e+01, 9.795494558942352e+01, 9.879062170560421e-03, 1.210441605771071e-02, 1.332449292266646e-02, 1.274225559008428e-02, 1.324625558489187e-02, 1.324625558489187e-02, 1.164461050505188e-02, -1.251211625294384e-02, -8.596870408407795e-03, -2.652099148853147e-03, -6.283971969526164e-03, -6.283971969526223e-03, 6.222266292712081e-03, -3.251669060101201e-01, -1.879003167438511e-01, -6.225903391934831e-02, -1.118995426810230e-01, -1.118995426810223e-01, -2.004515329684525e-02, 1.573188988587554e+01, 1.127303129914223e+01, -7.561316250658798e-02, 4.999345007139175e+00, 4.999345007139162e+00, 5.660186232868967e+01, 1.265833852934626e+02, 1.041716124680382e+02, 6.373177832586675e+00, 1.457804672846694e+02, 1.457804672846382e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05