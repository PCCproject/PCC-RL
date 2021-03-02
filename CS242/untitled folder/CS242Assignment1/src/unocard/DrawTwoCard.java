package unocard;
import game.SpecialEffectsManager;

public class DrawTwoCard extends ActionCard{
	
	public DrawTwoCard(Color color) {
		super(color, ActionCard.Content.DRAWTWO);
	}
	
	/**
	 * Next player in sequence draws two cards and misses a turn, unless they have 
	 * another Draw Two card to "stack" the number of cards to draw for the next player
	 * @param effManager: the object that controls the game effects.
	 */
	@Override
	public void doSpecialEffect(SpecialEffectsManager effManager){
		effManager.doDrawTwo();
	}
}
